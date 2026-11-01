extends Node3D
var _binary_client: RefCounted
var _sources: Dictionary = {}
var _volumes: Dictionary = {}
var _parity: Object
var agent_multi: MultiMeshInstance3D
var _link_layer: Object
var _budget: Object
var _memory_cloud: Object
var _beat: Object

func _tick_frame_budget(delta: float) -> void:
	if _binary_client == null:
		return
	var sum := {"gems": 0, "faded": 0, "halos": 0, "impostors": 0, "cylinders": 0, "ribbons": 0}
	var role_key := {"nodes": "gems", "faded": "faded", "halos": "halos", "impostors": "impostors", "edges": "cylinders", "ribbons": "ribbons"}
	var calls: int = 0
	var volumes := {}
	for vid: StringName in _sources:
		var src = _sources[vid]
		for role: String in role_key:
			var c: int = int(src.counts.get(role, 0))
			sum[role_key[role]] += c
			if c > 0:
				calls += 1
		var v: Node3D = _volumes.get(vid, null)
		# The volume's own frame (and handle markers) are draw calls outside the
		# node and edge instances; the extras' tier 2 ribbons reuse that call.
		if v != null and v.has_method("overhead_draw_calls"):
			calls += int(v.overhead_draw_calls())
		volumes[vid] = src.demand() if v != null and v.visible_state else {"nodes": 0, "edges": 0}
	var layers := sum.duplicate()
	layers["hulls"] = _parity.hull_count if _parity != null else 0
	layers["hull_tris"] = _parity.hull_triangles if _parity != null else 0
	if agent_multi != null and agent_multi.multimesh != null and agent_multi.visible and agent_multi.multimesh.instance_count > 0:
		calls += 1
	# §7.7: the link layer's call is charged here; its triangles are inside the
	# renderer's measured total, which the pass charges as outside-layer triangles.
	if _link_layer != null:
		calls += int(_link_layer.draw_calls())
	if int(layers["hulls"]) > 0:
		calls += 1
	layers["draw_calls"] = calls
	var measured: int = int(RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME))
	var bursts_on: bool = _beat != null and bool(_beat.get("bursts_enabled"))
	if _budget.tick(delta, _binary_client, layers, _memory_cloud, measured, bursts_on, volumes) and _parity != null:
		_parity.max_hulls = int(_budget.caps["max_hulls"])
