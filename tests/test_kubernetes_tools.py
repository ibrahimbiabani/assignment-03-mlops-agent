"""Kubernetes tool tests. The cluster is faked with the client's real model classes."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import urllib3
from kubernetes import client as k
from kubernetes.client.exceptions import ApiException

from agent.schemas import ToolStatus
from agent.tools import kubernetes_tools as kt

NS = "mlops-app"
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


# ---------- builders for real kubernetes.client model objects ----------


def _probe(path, period):
    return k.V1Probe(
        http_get=k.V1HTTPGetAction(path=path, port="http"),
        period_seconds=period,
        failure_threshold=3,
    )


def make_pod(name, phase="Running", ready=True, restarts=0, waiting=None, last_reason=None):
    if waiting:
        state = k.V1ContainerState(
            waiting=k.V1ContainerStateWaiting(reason=waiting, message="back-off restarting")
        )
    else:
        state = k.V1ContainerState(running=k.V1ContainerStateRunning(started_at=NOW))
    last = None
    if last_reason:
        last = k.V1ContainerState(
            terminated=k.V1ContainerStateTerminated(exit_code=137, reason=last_reason)
        )
    status = k.V1ContainerStatus(
        name="prediction-api",
        image="prediction-api:1.0.0",
        image_id="sha256:abc",
        ready=ready,
        restart_count=restarts,
        state=state,
        last_state=last,
    )
    container = k.V1Container(
        name="prediction-api",
        image="prediction-api:1.0.0",
        readiness_probe=_probe("/health/ready", 5),
        liveness_probe=_probe("/health/live", 10),
    )
    return k.V1Pod(
        metadata=k.V1ObjectMeta(name=name, labels={"app.kubernetes.io/version": "1.0.0"}),
        spec=k.V1PodSpec(node_name="docker-desktop", containers=[container]),
        status=k.V1PodStatus(
            phase=phase,
            start_time=NOW,
            container_statuses=[status],
            conditions=[k.V1PodCondition(type="Ready", status=str(ready))],
        ),
    )


def make_deployment(name="prediction-api", desired=2, ready=2):
    template = k.V1PodTemplateSpec(
        spec=k.V1PodSpec(containers=[k.V1Container(name=name, image="prediction-api:1.0.0")])
    )
    return k.V1Deployment(
        metadata=k.V1ObjectMeta(
            name=name,
            labels={"app.kubernetes.io/version": "1.0.0"},
            annotations={"deployment.kubernetes.io/revision": "3"},
        ),
        spec=k.V1DeploymentSpec(
            replicas=desired,
            selector=k.V1LabelSelector(match_labels={"app": name}),
            template=template,
        ),
        status=k.V1DeploymentStatus(
            ready_replicas=ready, available_replicas=ready, updated_replicas=desired
        ),
    )


def make_event(pod_name, reason, message, minutes_ago, type_="Warning"):
    return k.CoreV1Event(
        metadata=k.V1ObjectMeta(name=f"{pod_name}.{reason}"),
        involved_object=k.V1ObjectReference(kind="Pod", name=pod_name),
        reason=reason,
        message=message,
        type=type_,
        count=4,
        last_timestamp=NOW - timedelta(minutes=minutes_ago),
    )


class FakeCluster:
    """Implements the CoreV1/AppsV1/DiscoveryV1 methods the tools call, recording each call."""

    def __init__(self):
        self.pods = [make_pod("api-1"), make_pod("api-2")]
        self.deployments = [make_deployment()]
        self.events = []
        self.logs = b"INFO started\nERROR could not reach MLflow token=abc123\n"
        self.version_body = b'{"app_version": "1.0.0", "model": {"version": "1"}}'
        self.ready_endpoints = [True, True]
        self.fail: dict[str, Exception] = {}
        self.calls: list[tuple[str, tuple, dict]] = []

    def _call(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        if method in self.fail:
            raise self.fail[method]

    def list_namespaced_pod(self, ns, **kw):
        self._call("list_namespaced_pod", ns, **kw)
        return k.V1PodList(items=self.pods)

    def read_namespaced_pod(self, name, ns, **kw):
        self._call("read_namespaced_pod", name, ns, **kw)
        return next(p for p in self.pods if p.metadata.name == name)

    def list_namespaced_event(self, ns, **kw):
        self._call("list_namespaced_event", ns, **kw)
        return k.CoreV1EventList(items=self.events)

    def read_namespaced_pod_log(self, name, ns, **kw):
        self._call("read_namespaced_pod_log", name, ns, **kw)
        return SimpleNamespace(data=self.logs)

    def list_namespaced_service(self, ns, **kw):
        self._call("list_namespaced_service", ns, **kw)
        spec = k.V1ServiceSpec(
            type="ClusterIP",
            cluster_ip="10.0.0.5",
            selector={"app.kubernetes.io/name": "prediction-api"},
            ports=[k.V1ServicePort(name="http", port=80, target_port="http")],
        )
        return k.V1ServiceList(
            items=[k.V1Service(metadata=k.V1ObjectMeta(name="prediction-api"), spec=spec)]
        )

    def list_namespaced_endpoint_slice(self, ns, **kw):
        self._call("list_namespaced_endpoint_slice", ns, **kw)
        endpoints = [
            k.V1Endpoint(addresses=[f"10.1.0.{i}"], conditions=k.V1EndpointConditions(ready=r))
            for i, r in enumerate(self.ready_endpoints)
        ]
        return k.V1EndpointSliceList(
            items=[k.V1EndpointSlice(address_type="IPv4", endpoints=endpoints)]
        )

    def connect_get_namespaced_service_proxy_with_path(self, name, ns, path, **kw):
        self._call("service_proxy", name, ns, path, **kw)
        return SimpleNamespace(data=self.version_body)

    def list_namespaced_deployment(self, ns, **kw):
        self._call("list_namespaced_deployment", ns, **kw)
        return k.V1DeploymentList(items=self.deployments)

    def read_namespaced_deployment(self, name, ns, **kw):
        self._call("read_namespaced_deployment", name, ns, **kw)
        return self.deployments[0]

    def patch_namespaced_deployment(self, name, ns, body, **kw):
        self._call("patch_namespaced_deployment", name, ns, body, **kw)
        return self.deployments[0]

    def patch_namespaced_deployment_scale(self, name, ns, body, **kw):
        self._call("patch_namespaced_deployment_scale", name, ns, body, **kw)
        return k.V1Scale(spec=k.V1ScaleSpec(replicas=body["spec"]["replicas"]))


@pytest.fixture
def cluster(monkeypatch):
    fake = FakeCluster()
    monkeypatch.delenv("K8S_ALLOWED_NAMESPACES", raising=False)
    monkeypatch.setattr(
        kt, "get_apis", lambda: SimpleNamespace(core=fake, apps=fake, discovery=fake)
    )
    return fake


# ---------- read-only tools ----------


def test_list_workloads(cluster):
    result = kt.list_workloads()

    assert result.ok
    dep = result.data["deployments"][0]
    assert (dep["replicas_desired"], dep["replicas_ready"], dep["version"]) == (2, 2, "1.0.0")
    assert [p["name"] for p in result.data["pods"]] == ["api-1", "api-2"]
    assert all(call[2]["_request_timeout"] == 10.0 for call in cluster.calls)


def test_pod_status_reports_crash_loop_probes_and_events(cluster):
    cluster.pods = [
        make_pod(
            "api-1", ready=False, restarts=5, waiting="CrashLoopBackOff", last_reason="OOMKilled"
        )
    ]
    cluster.events = [make_event("api-1", "BackOff", "Back-off restarting failed container", 1)]

    result = kt.get_pod_status("api-1")

    container = result.data["containers"][0]
    assert container["state"] == {
        "waiting": {"reason": "CrashLoopBackOff", "message": "back-off restarting"}
    }
    assert container["last_state"]["terminated"]["reason"] == "OOMKilled"
    assert result.data["probes"]["prediction-api"]["readiness"]["path"] == "/health/ready"
    assert result.data["events"][0]["reason"] == "BackOff"
    event_call = next(c for c in cluster.calls if c[0] == "list_namespaced_event")
    assert event_call[2]["field_selector"] == "involvedObject.name=api-1"


def test_pod_logs_decoded_redacted_with_error_lines(cluster):
    result = kt.get_pod_logs("api-1", previous=True, tail_lines=50)

    assert result.data["lines"][0] == "INFO started"
    assert result.data["error_lines"] == ["ERROR could not reach MLflow token=***"]
    call = next(c for c in cluster.calls if c[0] == "read_namespaced_pod_log")
    assert call[2]["previous"] is True and call[2]["tail_lines"] == 50


def test_services_report_ready_endpoints(cluster):
    cluster.ready_endpoints = [True, False]
    svc = kt.list_services().data["services"][0]
    assert (svc["type"], svc["ready_endpoints"], svc["not_ready_endpoints"]) == ("ClusterIP", 1, 1)


def test_find_unhealthy_reports_pods_deployments_and_newest_warnings_first(cluster):
    cluster.pods = [
        make_pod("ok"),
        make_pod(
            "crashing", ready=False, restarts=3, waiting="CrashLoopBackOff", last_reason="Error"
        ),
        make_pod("not-ready", ready=False),
    ]
    cluster.deployments = [make_deployment(desired=3, ready=1)]
    cluster.events = [
        make_event("not-ready", "Unhealthy", "Readiness probe failed: HTTP 503", 10),
        make_event("crashing", "BackOff", "Back-off restarting failed container", 1),
    ]

    data = kt.find_unhealthy().data

    assert data["healthy"] is False
    reasons = {p["name"]: p["reasons"] for p in data["unhealthy_pods"]}
    assert reasons["crashing"] == [
        "not ready (readiness probe failing)",
        "waiting: CrashLoopBackOff",
    ]
    assert reasons["not-ready"] == ["not ready (readiness probe failing)"]
    assert "ok" not in reasons
    assert data["restarted_pods"][0]["name"] == "crashing"
    assert data["under_replicated_deployments"][0]["replicas_ready"] == 1
    assert [e["reason"] for e in data["warning_events"]] == ["BackOff", "Unhealthy"]


def test_healthy_namespace(cluster):
    assert kt.find_unhealthy().data["healthy"] is True


def test_clean_restart_is_reported_but_not_a_fault(cluster):
    clean = make_pod("rebooted", restarts=2)
    clean.status.container_statuses[0].last_state = k.V1ContainerState(
        terminated=k.V1ContainerStateTerminated(exit_code=0, reason="Completed")
    )
    cluster.pods = [clean, make_pod("oom", restarts=1, last_reason="OOMKilled")]

    data = kt.find_unhealthy().data
    crashed = {r["name"]: r["crashed"] for r in data["restarted_pods"]}
    assert crashed == {"rebooted": False, "oom": True}
    assert data["healthy"] is False  # because of the OOM kill

    cluster.pods = [clean]
    assert kt.find_unhealthy().data["healthy"] is True


def test_deployed_version_includes_what_the_app_reports(cluster):
    data = kt.get_deployed_version().data

    assert data["deployment"]["revision"] == "3"
    assert data["app_reported"] == {"app_version": "1.0.0", "model": {"version": "1"}}
    proxy = next(c for c in cluster.calls if c[0] == "service_proxy")
    assert proxy[1] == ("prediction-api:http", NS, "version")


def test_deployed_version_still_succeeds_when_app_unreachable(cluster):
    cluster.fail["service_proxy"] = ApiException(status=503, reason="Service Unavailable")

    result = kt.get_deployed_version()

    assert result.ok
    assert result.data["app_reported"] is None
    assert "503" in result.data["app_reported_error"]


# ---------- safety: confirmation and namespace restriction ----------


@pytest.mark.parametrize(
    "call",
    [
        lambda: kt.restart_deployment("prediction-api"),
        lambda: kt.scale_deployment("prediction-api", 3),
        lambda: kt.restart_deployment("prediction-api", confirm="yes"),
    ],
)
def test_modifications_without_confirmation_are_not_executed(cluster, call):
    result = call()
    assert result.status == ToolStatus.CONFIRMATION_REQUIRED
    assert "confirmation" in result.error
    assert cluster.calls == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: kt.restart_deployment("coredns", namespace="kube-system", confirm=True),
        lambda: kt.scale_deployment("web", 3, namespace="default", confirm=True),
        lambda: kt.list_workloads(namespace="kube-system"),
        lambda: kt.get_pod_logs("etcd-docker-desktop", namespace="kube-system"),
    ],
)
def test_unauthorized_namespace_is_rejected_even_when_confirmed(cluster, call):
    result = call()
    assert result.status == ToolStatus.REJECTED
    assert "not approved" in result.error
    assert cluster.calls == []


def test_confirmed_restart_patches_pod_template_annotation(cluster):
    result = kt.restart_deployment("prediction-api", confirm=True)

    assert result.ok
    name, ns, body = cluster.calls[-1][1]
    assert (name, ns) == ("prediction-api", NS)
    annotations = body["spec"]["template"]["metadata"]["annotations"]
    assert annotations["kubectl.kubernetes.io/restartedAt"] == result.data["restarted_at"]


def test_confirmed_scale(cluster):
    result = kt.scale_deployment("prediction-api", 3, confirm=True)
    assert result.data["replicas"] == 3
    assert cluster.calls[-1][1][2] == {"spec": {"replicas": 3}}


@pytest.mark.parametrize("replicas", [0, 6, True, "3", 2.5])
def test_scale_bounds(cluster, replicas):
    result = kt.scale_deployment("prediction-api", replicas, confirm=True)
    assert result.status == ToolStatus.INVALID_REQUEST
    assert cluster.calls == []


@pytest.mark.parametrize("name", ["Prediction_API", "../x", "", "a" * 300])
def test_invalid_names_rejected(cluster, name):
    assert kt.get_pod_status(name).status == ToolStatus.INVALID_REQUEST
    assert cluster.calls == []


# ---------- API and connectivity failures: no invented results ----------


@pytest.mark.parametrize(
    "exc, status",
    [
        (ApiException(status=404, reason="Not Found"), ToolStatus.INVALID_REQUEST),
        (ApiException(status=403, reason="Forbidden"), ToolStatus.REJECTED),
        (ApiException(status=500, reason="Internal Server Error"), ToolStatus.ERROR),
        (urllib3.exceptions.ReadTimeoutError(None, "/api", "read timed out"), ToolStatus.TIMEOUT),
        (
            urllib3.exceptions.MaxRetryError(
                None, "/api", urllib3.exceptions.NewConnectionError(None, "refused")
            ),
            ToolStatus.ERROR,
        ),
    ],
)
def test_api_failures_map_to_status_without_data(cluster, exc, status):
    cluster.fail["list_namespaced_pod"] = exc

    result = kt.find_unhealthy()

    assert result.status == status
    assert result.data is None


def test_missing_kubeconfig_is_reported_as_error(tmp_path, monkeypatch):
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "no-such-config"))
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    result = kt.list_workloads()

    assert result.status == ToolStatus.ERROR
    assert "configuration is not available" in result.error
