extends SceneTree
const Portals = preload("res://scripts/space_portals.gd")
const Arrangement = preload("res://scripts/volume_arrangement.gd")
const Volume = preload("res://scenes/EmulatedVolume.tscn")
var failures := 0
var cases: Array = []

func check(ok: bool, message: String) -> void:
	if not ok:
		failures += 1
		push_error(message)

func _initialize() -> void:
	call_deferred("run")

func fixture(extra := 2) -> Dictionary:
	var arr = Arrangement.new()
	root.add_child(arr)
	for i in range(extra + 1):
		var v = Volume.instantiate()
		v.volume_id = StringName("volume%d" % i)
		arr.add_volume(v, float(i) * 50.0)
	var p = Portals.new()
	p.bind(arr)
	p.walkthrough_enabled = true
	check(p.enter(&"volume0", Vector3(0, 1.6, 0)), "enter fixture")
	for i in 40:
		arr.tick(1.0 / 90.0)
		p.tick(1.0 / 90.0)
	return {"arr": arr, "p": p}

func run() -> void:
	# The same physical gap with a different arrangement scale changes re-arming.
	for row in [[1.0, 0.2, false], [0.5, 0.2, true], [1.0, 0.4, true], [2.0, 0.4, false]]:
		var f = fixture()
		var arr = f.arr
		var p = f.p
		var scale_factor: float = row[0]
		var world_gap: float = row[1]
		arr.scale = Vector3.ONE * scale_factor
		var v = arr.volume(&"volume1")
		var point: Vector3 = v.to_global(Vector3(0, 0, v.bounds.z * 0.5 + world_gap / scale_factor))
		var measured: float = Portals.distance_to_box(v, point)
		p.update_walkthrough(point, Vector3.FORWARD, 0.01)
		check(is_equal_approx(measured, world_gap / scale_factor), "distance in local units")
		check(p.is_walkthrough_armed() == row[2], "re-arm observation")
		cases.append({"case": "rearm_world_gap", "scale": scale_factor, "world_gap_m": world_gap,
			"helper_distance": measured, "armed": p.is_walkthrough_armed(), "expected_world_rule": world_gap > 0.3})
		arr.free()
	# Existing documented capacity is a control; registration itself has no cap.
	for count in [8, 9]:
		var f = fixture(count)
		var p = f.p
		check(p.portal_ids().size() == count, "all added volumes in portal list")
		check(p.membranes.multimesh.visible_instance_count == mini(count, 8), "membrane cap")
		cases.append({"case": "portal_capacity", "portal_boxes": p.portal_ids().size(),
			"membranes": p.membranes.multimesh.visible_instance_count,
			"classification": "control" if count == 8 else "out_of_documented_capacity"})
		f.arr.free()
	# A resize changes the real frame before the next tick begins the comfort dim.
	var f = fixture()
	var p = f.p
	var v = f.arr.volume(&"volume1")
	var before: Vector3 = v.bounds
	v.set_runtime_bounds(Vector3(2, 2, 2), 0)
	check(v.bounds != before and p.fade_alpha() == 0.0, "resize precedes dim")
	p.tick(1.0 / 90.0)
	check(p.is_transitioning() and p.fade_alpha() == 0.0, "dim starts after resize")
	cases.append({"case": "programmatic_resize_precedes_dim", "before": str(before),
		"after": str(v.bounds), "fade_alpha_after_detection": p.fade_alpha(),
		"classification": "API_level_counterexample_not_headset_observation"})
	f.arr.free()
	print("AUDIT_JSON " + JSON.stringify({"cases": cases, "failed_assertions": failures}))
	quit(0 if failures == 0 else 1)
