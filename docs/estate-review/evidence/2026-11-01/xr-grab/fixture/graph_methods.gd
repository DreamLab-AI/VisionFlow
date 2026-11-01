extends "res://spy_base.gd"

func _is_held(node_id: int) -> bool:
	return node_id >= 0 and (_grab_id[0] == node_id or _grab_id[1] == node_id)

func _held_ids(only_vol: StringName = &"") -> Array[int]:
	var out: Array[int] = []
	for hand: int in 2:
		if _grab_id[hand] != -1 and (only_vol == &"" or _grab_vol[hand] == only_vol):
			out.append(_grab_id[hand])
	return out

func _sync_grab_exclusion(vid: StringName) -> void:
	var src = _sources.get(vid, null)
	if src == null or not ("grabbed_ids" in src):
		return
	src.grabbed_ids = PackedInt32Array(_held_ids(vid))

func _release_grab(vid: StringName = &"") -> void:
	for hand: int in 2:
		if _grab_id[hand] != -1 and (vid == &"" or _grab_vol[hand] == vid):
			_end_grab(hand, false)

func _any_grab_id() -> int:
	return _grab_id[0] if _grab_id[0] != -1 else _grab_id[1]

static func volume_serves_drag(volume_id: StringName) -> bool:
	var b: Dictionary = VOLUME_BINDINGS.get(volume_id, {})
	if b.is_empty():
		return false
	return BackendCaps.supports(BackendCaps.for_mode(String(b["conn"])), BackendCaps.DRAG)

static func grab_volume_allowed(volume_id: StringName, entered: StringName) -> bool:
	return entered == &"" or volume_id == entered

func _grab_locked() -> bool:
	if _portals != null and _portals.is_transitioning():
		return true
	return _arrangement != null and _arrangement.is_easing()

func _grab_volume_of(node_id: int) -> StringName:
	var idx: int = _grab_eval_ids.find(node_id)
	if idx >= 0:
		for k: int in _grab_eval_ends.size():
			if idx < _grab_eval_ends[k]:
				return _grab_eval_vols[k]
	return _grab_eval_vol if _grab_eval_vol != &"" else OVERLAY_VOLUME

func _begin_grab(hand: int, vid: StringName, node_id: int, hand_pos: Vector3, hand_basis: Variant = null) -> bool:
	if hand < 0 or hand > 1 or node_id < 0 or _grab_id[hand] != -1 or _is_held(node_id):
		return false
	if not volume_serves_drag(vid) or not grab_volume_allowed(vid, _entered_volume()) or _grab_locked():
		return false
	var conn: Node = _conn_for_volume(vid)
	if conn == null:
		return false
	_grab_id[hand] = node_id
	_grab_vol[hand] = vid
	_sync_grab_exclusion(vid)
	print("GraphScene DEBUG: node_grabbed hand=%d volume=%s id=%d" % [hand, vid, node_id])
	# The held node is the selection (UX feature 2): the Rust tube may catch a
	# node the volume picker's cone missed, or a different one.
	select_volume_node(vid, StringName(VOLUME_BINDINGS[vid]["conn"]), node_id)
	# Node render position lives in the connection's Rust store.
	var pos: Vector3 = Vector3.ZERO
	if conn.client != null and conn.client.has_method("node_position"):
		pos = conn.client.node_position(node_id)
	_grab_target[hand] = pos
	# Remember how far along the ray the node was so the drag preserves depth.
	var node_world: Vector3 = _content_xf(vid) * pos
	_grab_dist[hand] = clampf(hand_pos.distance_to(node_world), 0.2, 6.0)
	_grab_lateral[hand] = Vector3.ZERO
	if hand_basis is Basis:
		# The catch offset in the hand's own frame: depth along -Z, lateral in x / y.
		var local: Vector3 = (hand_basis as Basis).orthonormalized().inverse() * (node_world - hand_pos)
		_grab_lateral[hand] = Vector3(local.x, local.y, 0.0)
		_grab_dist[hand] = clampf(-local.z, 0.2, 6.0)
	conn.send_drag_start(node_id, pos)
	_pulse(_ctrl_at(hand), 0.6, 0.08)
	return true

func _drive_grab(hand: int, ray_origin: Vector3, ray_dir: Vector3, hand_basis: Variant = null) -> void:
	var node_id: int = _grab_id[hand]
	if node_id == -1:
		return
	var vid: StringName = _grab_vol[hand]
	var hand_world: Vector3 = ray_origin + ray_dir * float(_grab_dist[hand])
	if hand_basis is Basis:
		hand_world += (hand_basis as Basis).orthonormalized() * (_grab_lateral[hand] as Vector3)
	var hand_server: Vector3 = _content_xf(vid).affine_inverse() * hand_world
	_grab_target[hand] = hand_server
	var conn: Node = _conn_for_volume(vid)
	if conn != null:
		conn.send_drag_update(node_id, hand_server)

func _end_grab(hand: int, pulse: bool = true) -> void:
	var node_id: int = _grab_id[hand]
	if node_id == -1:
		return
	var vid: StringName = _grab_vol[hand]
	var conn: Node = _conn_for_volume(vid)
	if conn != null:
		conn.send_drag_end(node_id)
	if pulse:
		_pulse(_ctrl_at(hand), 0.3, 0.05)
	_grab_id[hand] = -1
	_grab_vol[hand] = &""
	_grab_dist[hand] = 1.0
	_grab_lateral[hand] = Vector3.ZERO
	_grab_target[hand] = Vector3.ZERO
	# After the state is cleared, so the other hand's node (if it is in the same
	# volume) stays excluded and this one rejoins the fit.
	_sync_grab_exclusion(vid)

func _on_node_grabbed(node_id: int, _position: Vector3) -> void:
	# node_grabbed fires synchronously inside evaluate_ray, which names no hand:
	# _update_interaction records the evaluating hand and its volumes first.
	var hand: int = _grab_eval_hand
	if hand < 0 or hand > 1 or _grab_id[hand] == node_id or _is_held(node_id):
		return
	# Double-click = two grabs of the SAME node within the window → open the node's
	# linked page in the HUD instead of grabbing. The grab this second press would
	# start is suppressed cleanly (we return before the grab begins).
	var now: float = float(Time.get_ticks_msec()) / 1000.0
	if node_id == _dc_last_id[hand] and (now - _dc_last_time[hand]) <= DOUBLE_CLICK_SEC:
		_dc_last_id[hand] = -1
		_open_node_document(node_id)
		return
	_dc_last_id[hand] = node_id
	_dc_last_time[hand] = now
	var vid: StringName = _grab_volume_of(node_id)
	var ctrl: XRController3D = _ctrl_at(hand)
	var ctrl_basis: Variant = ctrl.global_transform.basis if ctrl != null else null
	_begin_grab(hand, vid, node_id, ctrl.global_position if ctrl != null else Vector3.ZERO, ctrl_basis)

func _on_conn_topology_changed(conn_id: StringName, _generation: int) -> void:
	for vid: StringName in _sources:
		if VOLUME_BINDINGS[vid]["conn"] == conn_id:
			_sources[vid].invalidate()
	if conn_id == &"visionclaw":
		_cloud_placed = false
	# Wire ids are per generation: a selection on this connection is void.
	_clear_selection_for_conn(conn_id)
	# §7.5: the old generation's links are void; the join reruns on the next
	# link tick against the new tables.
	if _link_layer != null:
		_link_layer.drop_connection(conn_id)
	_link_sig = []
	_link_join_accum = LINK_JOIN_SEC
