extends SceneTree
const Candy := preload("res://scripts/eye_candy.gd")
const Volume := preload("res://scenes/EmulatedVolume.tscn")
const SceneTick := preload("res://scene_tick.gd")
var results: Array = []

class Capture:
	extends RefCounted
	var last: Dictionary
	func tick(_d, _c, layers, _m, _t, _b, _v):
		last = layers
		return false

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var c := Candy.new()
	c._governor = Candy.Governor.gpu_mode()
	c._governor.tier = 4
	c.signal_mode = "gpu"
	# 40 seconds, one loaded GPU result every 180 frames. Every other sample is absent.
	for i: int in 3600:
		c._feed_gpu(12.0 if i % 180 == 179 else 0.0, 1.0 / 90.0)
	results.append({"case": "sparse_loaded_gpu", "wall_seconds": 40, "positive_samples": 20,
		"mode": c.signal_mode, "tier": c.wanted_tier(), "governor_elapsed": c._governor._elapsed})
	assert(c.signal_mode == "gpu" and c.wanted_tier() == 4)
	for i: int in 181:
		c._feed_gpu(0.0, 1.0 / 90.0)
	results.append({"case": "consecutive_zero_control", "mode": c.signal_mode, "tier": c.wanted_tier()})
	assert(c.signal_mode == "frame" and c.wanted_tier() == 2)
	c._forced_tier = 4
	results.append({"case": "manual_pin_bypasses_fallback_ceiling", "mode": c.signal_mode, "requested": c.wanted_tier()})
	assert(c.wanted_tier() == 4)
	c.free()

	var scene := Node3D.new()
	root.add_child(scene)
	var candy := Candy.new()
	scene.add_child(candy)
	candy.set_process(false)
	candy.setup(scene)
	var built: int = 0
	for i: int in 9:
		var v := Volume.instantiate()
		scene.add_child(v)
	candy.apply_tier(2)
	for v: Node in get_nodes_in_group("xr_volume_frames"):
		built += v.frame_triangles()
	results.append({"case": "nine_visible_frames", "frames": candy._visible_frames(),
		"built_triangles": built, "reported_triangles": candy.applied_cost().x})
	assert(candy._visible_frames() == 9 and built == 216 and candy.applied_cost().x == 192)
	scene.free()

	var tick := SceneTick.new()
	root.add_child(tick)
	tick._binary_client = RefCounted.new()
	var capture := Capture.new()
	tick._budget = capture
	tick._tick_frame_budget(0.016)
	var before: int = capture.last.draw_calls
	# Real extra mesh nodes; dummy renderer gives no measured call count.
	# The observation is the demand passed by the unchanged scene method.
	for i: int in 20:
		var mesh := MeshInstance3D.new()
		mesh.mesh = BoxMesh.new()
		tick.add_child(mesh)
	tick._tick_frame_budget(0.016)
	results.append({"case": "unlisted_scene_meshes", "extra_mesh_nodes": 20,
		"demand_calls_before": before, "demand_calls_after": capture.last.draw_calls,
		"scope": "exact scene method; headless; renderer calls not measured"})
	assert(before == 0 and int(capture.last.draw_calls) == 0)
	tick.free()
	var output := FileAccess.open("res://probe-results.json", FileAccess.WRITE)
	output.store_string(JSON.stringify(results, "  "))
	output.close()
	print("AUDIT: five observations verified")
	quit(0)
