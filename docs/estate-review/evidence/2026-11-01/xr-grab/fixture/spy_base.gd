extends Node3D
# Audit scaffolding only. Every replacement below is explicit; graph_methods.gd
# contains unchanged production functions. No live transport or UI is exercised.
const BackendCaps = preload("res://backend_caps.gd")
const OVERLAY_VOLUME := &"knowledge"
const VOLUME_BINDINGS := {
	&"knowledge": {"conn": &"visionclaw"},
	&"ontology": {"conn": &"visionclaw"},
	&"codescape": {"conn": &"codescape"},
}
const DOUBLE_CLICK_SEC := 0.45
const LINK_JOIN_SEC := 1.0
var _grab_id: Array[int] = [-1, -1]
var _grab_vol: Array[StringName] = [&"", &""]
var _grab_dist: Array[float] = [1.0, 1.0]
var _grab_lateral: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO]
var _grab_target: Array[Vector3] = [Vector3.ZERO, Vector3.ZERO]
var _grab_eval_hand := 0
var _grab_eval_vol := &"knowledge"
var _grab_eval_ids := PackedInt32Array()
var _grab_eval_vols: Array[StringName] = []
var _grab_eval_ends := PackedInt32Array()
var _dc_last_id: Array[int] = [-1, -1]
var _dc_last_time: Array[float] = [-100.0, -100.0]
var _sources: Dictionary = {}
var _cloud_placed := true
var _link_layer: Node = null
var _link_sig: Array = []
var _link_join_accum := 0.0
var _box_grab_ctrl: Node3D = null
var _box_grab_vid := &""
var _portals: Node = null
var _arrangement: Node = null
var selected_clears := 0
var opened: Array[int] = []
var xf := Transform3D.IDENTITY

class SpyClient:
	extends RefCounted
	var positions: Dictionary = {11: Vector3(0, 0, -1), 22: Vector3(0, 0, -2)}
	func node_position(id: int) -> Vector3:
		return positions.get(id, Vector3.ZERO)

class SpyConnection:
	extends Node
	var client = SpyClient.new()
	var calls: Array = []
	var admit := true
	func send_drag_start(id: int, p: Vector3) -> bool:
		calls.append(["start", id, p, admit])
		return admit
	func send_drag_update(id: int, p: Vector3) -> bool:
		calls.append(["update", id, p, admit])
		return admit
	func send_drag_end(id: int) -> bool:
		calls.append(["end", id, admit])
		return admit

class SpySource:
	extends RefCounted
	var grabbed_ids := PackedInt32Array()
	var invalidations := 0
	func invalidate() -> void:
		invalidations += 1

class SpyArrangement:
	extends Node
	var easing := false
	func is_easing() -> bool:
		return easing

var conn = SpyConnection.new()

func _init() -> void:
	_sources[&"knowledge"] = SpySource.new()
	_sources[&"ontology"] = SpySource.new()
	_arrangement = SpyArrangement.new()
	add_child(conn)
	add_child(_arrangement)

func _conn_for_volume(_vid: StringName) -> Node:
	return conn

func _content_xf(_vid: StringName) -> Transform3D:
	return xf

func _entered_volume() -> StringName:
	return &""

func _ctrl_at(_hand: int) -> XRController3D:
	return null

func select_volume_node(_vid: StringName, _conn: StringName, _id: int) -> void:
	pass # UI spy boundary: selection rendering is outside this fixture.

func _pulse(_controller, _amplitude: float, _seconds: float) -> void:
	pass # No controller or haptic device is present.

func _open_node_document(id: int) -> void:
	opened.append(id)

func _clear_selection_for_conn(_conn: StringName) -> void:
	selected_clears += 1
