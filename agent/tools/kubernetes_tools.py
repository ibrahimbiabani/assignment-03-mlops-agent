"""Kubernetes operations tools (Task 4), via the official Kubernetes Python client.

Safety:
* Every tool is limited to the namespaces in K8S_ALLOWED_NAMESPACES (default: mlops-app);
  the check happens before any API call.
* Only k8s_restart_deployment and k8s_scale_deployment change cluster state. Both also
  require confirm=True. Nothing can create or delete resources.
* The kubeconfig context is K8S_CONTEXT (default: docker-desktop). Point it at the
  restricted `mlops-agent` context (kubernetes/agent-rbac.yaml) so the cluster itself
  enforces the same limits.
"""

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import urllib3
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException

from agent.tools.base import InvalidToolRequest, ToolError, ToolRejected, ToolTimeout, tool

DEFAULT_NAMESPACE = "mlops-app"
MAX_REPLICAS = 5
_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?$")
_ERROR_LINE_RE = re.compile(r"error|exception|traceback|failed|refused|denied", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|token|secret|api[_-]?key|authorization)\b(\s*[=:]\s*)(\S+)"
)


def allowed_namespaces() -> frozenset[str]:
    raw = os.environ.get("K8S_ALLOWED_NAMESPACES", DEFAULT_NAMESPACE)
    return frozenset(n.strip() for n in raw.split(",") if n.strip())


def _timeout() -> float:
    return float(os.environ.get("K8S_TIMEOUT_S", "10"))


def get_apis() -> SimpleNamespace:
    """Kubernetes API clients for K8S_CONTEXT (or in-cluster config inside a Pod)."""
    try:
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            config.load_incluster_config()
        else:
            # Read KUBECONFIG now: the client only reads it once, at import time.
            config.load_kube_config(
                config_file=os.environ.get("KUBECONFIG"),
                context=os.environ.get("K8S_CONTEXT", "docker-desktop"),
            )
    except (ConfigException, FileNotFoundError, TypeError) as exc:
        raise ToolError(f"Kubernetes configuration is not available: {exc}") from exc
    api = client.ApiClient()
    return SimpleNamespace(
        core=client.CoreV1Api(api), apps=client.AppsV1Api(api), discovery=client.DiscoveryV1Api(api)
    )


@contextmanager
def _k8s_errors() -> Iterator[None]:
    try:
        yield
    except ApiException as exc:
        reason = f"HTTP {exc.status} {exc.reason}"
        if exc.status == 404:
            raise InvalidToolRequest(f"not found in Kubernetes ({reason})") from exc
        if exc.status == 403:
            raise ToolRejected(f"Kubernetes RBAC denied the operation ({reason})") from exc
        if exc.status in (400, 422):
            raise InvalidToolRequest(f"Kubernetes rejected the request ({reason})") from exc
        raise ToolError(f"Kubernetes API error ({reason})") from exc
    # NewConnectionError subclasses ConnectTimeoutError in urllib3, so test it first:
    # a refused connection means "unreachable", not "slow".
    except urllib3.exceptions.NewConnectionError as exc:
        raise ToolError("Kubernetes API server is unreachable") from exc
    except (urllib3.exceptions.TimeoutError, TimeoutError) as exc:
        raise ToolTimeout(f"Kubernetes API did not respond within {_timeout()}s") from exc
    except urllib3.exceptions.MaxRetryError as exc:
        reason = exc.reason
        if isinstance(reason, urllib3.exceptions.TimeoutError) and not isinstance(
            reason, urllib3.exceptions.NewConnectionError
        ):
            raise ToolTimeout(f"Kubernetes API did not respond within {_timeout()}s") from exc
        raise ToolError("Kubernetes API server is unreachable") from exc
    except urllib3.exceptions.HTTPError as exc:
        raise ToolError("Kubernetes API server is unreachable") from exc


def _namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not _NAME_RE.match(namespace):
        raise InvalidToolRequest(f"invalid namespace: {namespace!r}")
    if namespace not in allowed_namespaces():
        raise ToolRejected(
            f"namespace '{namespace}' is not approved; allowed: {sorted(allowed_namespaces())}"
        )
    return namespace


def _name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise InvalidToolRequest(f"invalid {kind} name: {value!r}")
    return value


def _redact(line: str) -> str:
    return _SECRET_RE.sub(r"\1\2***", line)


def _iso(ts: Any) -> str | None:
    return ts.isoformat() if ts else None


# ---------- summaries ----------


def _state(s: Any) -> dict[str, Any] | None:
    if s is None:
        return None
    if s.waiting:
        return {"waiting": {"reason": s.waiting.reason, "message": s.waiting.message}}
    if s.terminated:
        t = s.terminated
        return {
            "terminated": {
                "reason": t.reason,
                "exit_code": t.exit_code,
                "message": t.message,
                "finished_at": _iso(t.finished_at),
            }
        }
    if s.running:
        return {"running": {"started_at": _iso(s.running.started_at)}}
    return None


def _pod_summary(pod: Any) -> dict[str, Any]:
    statuses = pod.status.container_statuses or []
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": bool(statuses) and all(cs.ready for cs in statuses),
        "restarts": sum(cs.restart_count or 0 for cs in statuses),
        "node": pod.spec.node_name,
        "started": _iso(pod.status.start_time),
        "version": (pod.metadata.labels or {}).get("app.kubernetes.io/version"),
        "containers": [
            {
                "name": cs.name,
                "image": cs.image,
                "ready": cs.ready,
                "restart_count": cs.restart_count,
                "state": _state(cs.state),
                "last_state": _state(cs.last_state),
            }
            for cs in statuses
        ],
    }


def _deployment_summary(d: Any) -> dict[str, Any]:
    st = d.status
    return {
        "name": d.metadata.name,
        "version": (d.metadata.labels or {}).get("app.kubernetes.io/version"),
        "revision": (d.metadata.annotations or {}).get("deployment.kubernetes.io/revision"),
        "images": [c.image for c in d.spec.template.spec.containers],
        "replicas_desired": d.spec.replicas,
        "replicas_ready": st.ready_replicas or 0,
        "replicas_available": st.available_replicas or 0,
        "replicas_updated": st.updated_replicas or 0,
        "conditions": [
            {"type": c.type, "status": c.status, "reason": c.reason, "message": c.message}
            for c in st.conditions or []
        ],
    }


def _event(e: Any) -> dict[str, Any]:
    last = e.last_timestamp or e.event_time or e.metadata.creation_timestamp
    return {
        "type": e.type,
        "reason": e.reason,
        "object": f"{e.involved_object.kind}/{e.involved_object.name}",
        "message": e.message,
        "count": e.count,
        "last_seen": _iso(last),
    }


def _sorted_events(items: list[Any], limit: int) -> list[dict[str, Any]]:
    events = [_event(e) for e in items]
    events.sort(key=lambda e: e["last_seen"] or "", reverse=True)
    return events[:limit]


def _probe(p: Any) -> dict[str, Any] | None:
    if p is None:
        return None
    http = p.http_get
    return {
        "path": http.path if http else None,
        "port": http.port if http else None,
        "period_s": p.period_seconds,
        "failure_threshold": p.failure_threshold,
    }


# ---------- read-only tools ----------


@tool("k8s_list_workloads")
def list_workloads(namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
    """Deployments (desired/ready replicas, image, version) and Pods in the namespace."""
    ns = _namespace(namespace)
    apis, t = get_apis(), _timeout()
    with _k8s_errors():
        deployments = apis.apps.list_namespaced_deployment(ns, _request_timeout=t).items
        pods = apis.core.list_namespaced_pod(ns, _request_timeout=t).items
    return {
        "namespace": ns,
        "deployments": [_deployment_summary(d) for d in deployments],
        "pods": [_pod_summary(p) for p in pods],
    }


@tool("k8s_get_pod_status")
def get_pod_status(pod_name: str, namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
    """One Pod's phase, conditions, container states, probe settings and recent events."""
    ns, name = _namespace(namespace), _name(pod_name, "pod")
    apis, t = get_apis(), _timeout()
    with _k8s_errors():
        pod = apis.core.read_namespaced_pod(name, ns, _request_timeout=t)
        events = apis.core.list_namespaced_event(
            ns, field_selector=f"involvedObject.name={name}", _request_timeout=t
        ).items
    return {
        **_pod_summary(pod),
        "conditions": {c.type: c.status for c in pod.status.conditions or []},
        "probes": {
            c.name: {"readiness": _probe(c.readiness_probe), "liveness": _probe(c.liveness_probe)}
            for c in pod.spec.containers
        },
        "events": _sorted_events(events, 15),
    }


@tool("k8s_get_pod_logs")
def get_pod_logs(
    pod_name: str,
    namespace: str = DEFAULT_NAMESPACE,
    container: str | None = None,
    previous: bool = False,
    tail_lines: int = 100,
) -> dict[str, Any]:
    """Recent log lines of a Pod (previous=True: the last crashed container), secrets redacted."""
    ns, name = _namespace(namespace), _name(pod_name, "pod")
    if container is not None:
        _name(container, "container")
    if not isinstance(tail_lines, int) or not 1 <= tail_lines <= 1000:
        raise InvalidToolRequest("tail_lines must be an integer between 1 and 1000")
    apis = get_apis()
    with _k8s_errors():
        # Raw response: the client's own decoding returns str(bytes) ("b'...'") for logs.
        resp = apis.core.read_namespaced_pod_log(
            name,
            ns,
            container=container,
            previous=bool(previous),
            tail_lines=tail_lines,
            _preload_content=False,
            _request_timeout=_timeout(),
        )
    text = resp.data.decode("utf-8", errors="replace")
    lines = [_redact(ln)[:1000] for ln in text.splitlines()]
    return {
        "pod": name,
        "namespace": ns,
        "previous": bool(previous),
        "lines": lines,
        "error_lines": [ln for ln in lines if _ERROR_LINE_RE.search(ln)][-50:],
    }


@tool("k8s_list_services")
def list_services(namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
    """Services with type, ports, selector and how many ready Pods back each one."""
    ns = _namespace(namespace)
    apis, t = get_apis(), _timeout()
    out = []
    with _k8s_errors():
        for s in apis.core.list_namespaced_service(ns, _request_timeout=t).items:
            slices = apis.discovery.list_namespaced_endpoint_slice(
                ns,
                label_selector=f"kubernetes.io/service-name={s.metadata.name}",
                _request_timeout=t,
            ).items
            endpoints = [ep for sl in slices for ep in sl.endpoints or []]
            ready = sum(1 for ep in endpoints if ep.conditions and ep.conditions.ready)
            out.append(
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": [
                        {"name": p.name, "port": p.port, "target_port": str(p.target_port)}
                        for p in s.spec.ports or []
                    ],
                    "selector": s.spec.selector,
                    "ready_endpoints": ready,
                    "not_ready_endpoints": len(endpoints) - ready,
                }
            )
    return {"namespace": ns, "services": out}


@tool("k8s_find_unhealthy")
def find_unhealthy(namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
    """Unhealthy or restarting Pods, under-replicated Deployments and recent Warning events."""
    ns = _namespace(namespace)
    apis, t = get_apis(), _timeout()
    with _k8s_errors():
        pods = apis.core.list_namespaced_pod(ns, _request_timeout=t).items
        deployments = apis.apps.list_namespaced_deployment(ns, _request_timeout=t).items
        warnings = apis.core.list_namespaced_event(
            ns, field_selector="type=Warning", _request_timeout=t
        ).items

    unhealthy, restarted = [], []
    for p in pods:
        s = _pod_summary(p)
        waiting = [
            c["state"]["waiting"]["reason"]
            for c in s["containers"]
            if c["state"] and "waiting" in c["state"]
        ]
        reasons = []
        if s["phase"] not in ("Running", "Succeeded"):
            reasons.append(f"phase {s['phase']}")
        if s["phase"] == "Running" and not s["ready"]:
            reasons.append("not ready (readiness probe failing)")
        reasons += [f"waiting: {w}" for w in waiting]  # e.g. CrashLoopBackOff, ImagePullBackOff
        if reasons:
            unhealthy.append({**s, "reasons": reasons})
        if s["restarts"]:
            last = [c["last_state"] for c in s["containers"] if c["last_state"]]
            restarted.append({"name": s["name"], "restarts": s["restarts"], "last_states": last})
    under = [
        _deployment_summary(d)
        for d in deployments
        if (d.status.ready_replicas or 0) < (d.spec.replicas or 0)
    ]
    return {
        "namespace": ns,
        "healthy": not (unhealthy or restarted or under),
        "unhealthy_pods": unhealthy,
        "restarted_pods": restarted,
        "under_replicated_deployments": under,
        "warning_events": _sorted_events(warnings, 20),
    }


@tool("k8s_get_deployed_version")
def get_deployed_version(
    deployment: str = "prediction-api",
    namespace: str = DEFAULT_NAMESPACE,
    service: str | None = None,
) -> dict[str, Any]:
    """Deployed image/version label/revision, plus the model version the running app reports.

    The app's /version is read through the API server's service proxy, so it reflects
    what the Pods actually serve, not only what the manifest asked for.
    """
    ns, name = _namespace(namespace), _name(deployment, "deployment")
    svc = _name(service or deployment, "service")
    apis, t = get_apis(), _timeout()
    with _k8s_errors():
        d = apis.apps.read_namespaced_deployment(name, ns, _request_timeout=t)
    result = {"deployment": _deployment_summary(d)}
    try:
        with _k8s_errors():
            raw = apis.core.connect_get_namespaced_service_proxy_with_path(
                f"{svc}:http", ns, "version", _preload_content=False, _request_timeout=t
            )
        result["app_reported"] = json.loads(raw.data.decode())
    except (ToolError, ValueError) as exc:
        result["app_reported"] = None
        result["app_reported_error"] = str(exc)
    return result


# ---------- state-changing tools (confirmation required) ----------


def _require_confirmation(confirm: bool, action: str) -> None:
    if confirm is not True:
        raise ToolRejected(
            f"{action} changes cluster state and requires explicit user confirmation"
        )


@tool("k8s_restart_deployment")
def restart_deployment(
    deployment: str, namespace: str = DEFAULT_NAMESPACE, confirm: bool = False
) -> dict[str, Any]:
    """Rolling restart (like `kubectl rollout restart`). Requires confirm=True."""
    ns, name = _namespace(namespace), _name(deployment, "deployment")
    _require_confirmation(confirm, f"restarting deployment '{name}'")
    now = datetime.now(UTC).isoformat()
    body = {
        "spec": {
            "template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}}
        }
    }
    apis = get_apis()
    with _k8s_errors():
        d = apis.apps.patch_namespaced_deployment(name, ns, body, _request_timeout=_timeout())
    return {
        "namespace": ns,
        "deployment": name,
        "restarted_at": now,
        "status": _deployment_summary(d),
    }


@tool("k8s_scale_deployment")
def scale_deployment(
    deployment: str, replicas: int, namespace: str = DEFAULT_NAMESPACE, confirm: bool = False
) -> dict[str, Any]:
    """Set a Deployment's replica count (1..MAX_REPLICAS). Requires confirm=True."""
    ns, name = _namespace(namespace), _name(deployment, "deployment")
    if (
        not isinstance(replicas, int)
        or isinstance(replicas, bool)
        or not 1 <= replicas <= MAX_REPLICAS
    ):
        raise InvalidToolRequest(f"replicas must be an integer between 1 and {MAX_REPLICAS}")
    _require_confirmation(confirm, f"scaling deployment '{name}' to {replicas}")
    apis = get_apis()
    with _k8s_errors():
        scale = apis.apps.patch_namespaced_deployment_scale(
            name, ns, {"spec": {"replicas": replicas}}, _request_timeout=_timeout()
        )
    return {"namespace": ns, "deployment": name, "replicas": scale.spec.replicas}
