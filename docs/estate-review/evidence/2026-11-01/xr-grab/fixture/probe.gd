extends SceneTree
const Methods = preload("res://graph_methods.gd")
var failures := 0
var checks := 0
var results: Array = []

func check(condition: bool, label: String) -> void:
	checks += 1
	if not condition:
		failures += 1
		push_error(label)

func _initialize() -> void:
	# Control: independent hands and per-volume exclusion bookkeeping.
	var s = Methods.new()
	check(s._begin_grab(0, &"knowledge", 11, Vector3.ZERO), "left begins")
	check(s._begin_grab(1, &"ontology", 22, Vector3.ZERO), "right begins")
	check(not s._begin_grab(1, &"knowledge", 11, Vector3.ZERO), "held id refused")
	s._end_grab(0, false)
	check(s._grab_id[1] == 22, "right survives left release")
	check(s._sources[&"knowledge"].grabbed_ids.is_empty(), "left exclusion clears")
	check(s._sources[&"ontology"].grabbed_ids == PackedInt32Array([22]), "right exclusion survives")
	s._release_grab()
	check(s._grab_id == [-1, -1], "release all clears both")
	s.free()

	# Actual callback and drive bodies. Source and link objects are spies.
	s = Methods.new()
	check(s._begin_grab(0, &"knowledge", 11, Vector3.ZERO), "generation fixture begins")
	s.conn.calls.clear()
	s._on_conn_topology_changed(&"visionclaw", 2)
	check(s.selected_clears == 1, "topology clears selection control")
	check(s._sources[&"knowledge"].invalidations == 1, "topology invalidates source control")
	check(s._grab_id[0] == 11, "counterexample: held id survives generation change")
	s.conn.client.positions[11] = Vector3(10, 0, 0) # model a new id mapping
	s._drive_grab(0, Vector3.ZERO, Vector3.FORWARD)
	check(s.conn.calls.size() == 1 and s.conn.calls[0][0] == "update", "stale hold still sends update")
	results.append({"case": "generation", "held_id": s._grab_id[0], "sent_id": s.conn.calls[0][1], "sent_position": str(s.conn.calls[0][2]), "boundary": "new mapping is modelled, no server"})
	s.free()

	# evaluate_ray emits every active frame. Replaying its callback is enough to
	# inspect the consumer's absent press latch; no controller loop is claimed.
	s = Methods.new()
	s._on_node_grabbed(11, Vector3.ZERO)
	s._end_grab(0, false)
	s._on_node_grabbed(11, Vector3.ZERO)
	check(s.opened == [11] and s._grab_id[0] == -1, "second press opens instead of grabbing")
	s._on_node_grabbed(11, Vector3.ZERO) # same held trigger, next evaluate frame
	check(s._grab_id[0] == 11, "counterexample: next callback grabs after document open")
	results.append({"case": "double_click_continued_trigger", "opened": s.opened.duplicate(), "held_after_next_callback": s._grab_id[0]})
	s.free()

	# World-space depth controls and counterexamples at the capture limits.
	for depth: float in [0.1, 1.0, 6.0, 10.0]:
		s = Methods.new()
		s.conn.client.positions[11] = Vector3(0, 0, -depth)
		check(s._begin_grab(0, &"knowledge", 11, Vector3.ZERO, Basis.IDENTITY), "depth begins")
		s._drive_grab(0, Vector3.ZERO, Vector3.FORWARD, Basis.IDENTITY)
		var actual: Vector3 = s._grab_target[0]
		check(is_equal_approx(-actual.z, clampf(depth, 0.2, 6.0)), "depth clamp observed")
		results.append({"case": "depth", "initial_metres": depth, "first_drive_metres": -actual.z, "jump_metres": absf(depth + actual.z)})
		s.free()

	# Existing box-grip state is not part of _grab_locked. This tests node-start
	# admission only, not an actual box/controller gesture.
	s = Methods.new()
	var box_controller := Node3D.new()
	s._box_grab_ctrl = box_controller
	s._box_grab_vid = &"knowledge"
	check(s._begin_grab(0, &"knowledge", 11, Vector3.ZERO), "counterexample: node admitted during existing box grip")
	results.append({"case": "box_first", "node_admitted": s._grab_id[0] == 11})
	s.free()
	box_controller.free()

	# Positive lock control and an ignored send refusal.
	s = Methods.new()
	s._arrangement.easing = true
	check(not s._begin_grab(0, &"knowledge", 11, Vector3.ZERO), "ease blocks node start")
	s._arrangement.easing = false
	s.conn.admit = false
	check(s._begin_grab(0, &"knowledge", 11, Vector3.ZERO), "counterexample: false start send leaves local hold")
	results.append({"case": "send_refused", "local_held_id": s._grab_id[0], "send_return": s.conn.admit})
	s.free()
	print("AUDIT_RESULT " + JSON.stringify({"checks": checks, "failures": failures, "results": results}))
	quit(0 if failures == 0 else 1)
