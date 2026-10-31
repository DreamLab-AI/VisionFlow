extends SceneTree
const LinkLayer := preload("res://link_layer.gd")
const SpatialVolume := preload("res://spatial_volume.gd")
const XrPrefs := preload("res://xr_prefs.gd")

class Volume extends RefCounted:
	var visible_state := true
	var is_closed := false
	func world_from_content(p: Vector3) -> Vector3:
		return p

class Conn extends RefCounted:
	var generation := 1
	func node_position(id: int) -> Vector3:
		return Vector3(float(id), 0, 0)
	func is_connected_now() -> bool:
		return true

class Source extends RefCounted:
	var box := AABB(Vector3(-1,-1,-1), Vector3(2,2,2))
	func mount(_root: Node3D) -> void:
		pass
	func unmount() -> void:
		pass
	func content_aabb() -> AABB:
		return box

var cases: Array = []
var failures := 0

func check(ok: bool, label: String) -> void:
	if not ok:
		failures += 1
		push_error(label)

func rec(id: int, kind := "annotates") -> Dictionary:
	return {"source":{"volume":"codescape","conn":"codescape" if kind == "annotates" else "visionclaw","key":"source","id":1},
		"target":{"volume":"ontology","conn":"visionclaw","key":"t%04d" % id,"id":id},
		"kind":kind,"provenance":{"generations":[1,1]}}

func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var layer: Node3D = LinkLayer.new()
	root.add_child(layer)
	var volumes := {&"codescape":Volume.new(), &"ontology":Volume.new()}
	var conns := {&"codescape":Conn.new(), &"visionclaw":Conn.new()}
	var records: Array = []
	for i: int in 512:
		records.append(rec(i+10))
	layer.set_records("annotates",records)
	layer.set_enabled(true)
	layer.tick(0.0,volumes,conns)
	check(layer.multimesh_instance.multimesh.instance_count == 512, "plain cap")
	cases.append({"case":"plain_links_fit_budget","instances":512,"triangles":layer.triangle_count()})
	layer.set_highlight([{"conn":&"codescape","node_id":1}])
	var n: int = layer.multimesh_instance.multimesh.instance_count
	check(n == 1025 and layer.marker_count() == 513, "highlight counterexample")
	check(layer.triangle_count() == 12300 and layer.draw_calls() == 1, "marker cost is charged")
	cases.append({"case":"single_highlight_exceeds_instance_and_triangle_contract","instances":n,"markers":layer.marker_count(),"triangles":layer.triangle_count(),"draw_calls":layer.draw_calls(),"overflow":layer.overflow_count()})
	layer.set_highlight([])
	layer.set_records("annotates",[rec(10)])
	layer.set_records("bridge",[rec(11,"bridge")])
	layer.tick(0.0,volumes,conns)
	layer.set_enabled(false)
	check(layer.multimesh_instance.multimesh.instance_count == 0, "off clears immediately")
	layer.drop_connection(&"codescape")
	check(layer.multimesh_instance.multimesh.instance_count == 0, "off drop retains zero geometry")
	check(not layer.visible and layer.triangle_count() == 0, "off geometry remains hidden")
	cases.append({"case":"off_disconnect_keeps_zero_geometry","instances":layer.multimesh_instance.multimesh.instance_count,"visible":layer.visible,"reported_triangles":layer.triangle_count()})
	layer.set_enabled(true)
	layer.tick(0.0,volumes,conns)
	conns[&"visionclaw"].generation = 2
	layer.tick(1.0,volumes,conns)
	check(layer.record_count() == 0, "generation purge")
	cases.append({"case":"generation_change_purges_records","records":layer.record_count()})
	check(is_equal_approx(SpatialVolume.fit_scale(Vector3(1,2,3),Vector3(4,1,1)),0.25),"anisotropic fit")
	check(is_equal_approx(SpatialVolume.fit_scale(Vector3.ONE,Vector3.ZERO),1.0),"zero extent")
	cases.append({"case":"anisotropic_and_zero_extent_fit_controls"})
	var volume: Node3D = SpatialVolume.new()
	root.add_child(volume)
	var source := Source.new()
	volume.attach(source)
	volume.tick(1.0/90.0)
	check(volume.contains_world(volume.world_from_content(Vector3.ONE)),"initial fit")
	source.box = AABB(Vector3(-4,-4,-4),Vector3(8,8,8))
	var outside := 0
	for i: int in 5:
		volume.tick(1.0/90.0)
		if not volume.contains_world(volume.world_from_content(Vector3(4,4,4))):
			outside += 1
	check(outside == 4,"live extent waits for cadence")
	cases.append({"case":"growing_live_extent_outside_until_refit","outside_ticks":outside,"ticks_checked":5})
	volume.request_bounds(Vector3(0.5,0.5,0.5))
	volume.tick(1.0/90.0)
	check(volume.contains_world(volume.world_from_content(Vector3(4,4,4))),"resize refits immediately")
	cases.append({"case":"explicit_resize_refits_immediately"})
	var pref_path := "user://audit-prefs.cfg"
	check(XrPrefs._read(true,pref_path) and not XrPrefs._read(false,pref_path),"build defaults")
	check(XrPrefs._write(true,false,pref_path) == OK,"debug preference write")
	check(XrPrefs._write(false,true,pref_path) == OK,"release preference write")
	check(not XrPrefs._read(true,pref_path) and XrPrefs._read(false,pref_path),"build keys independent")
	DirAccess.remove_absolute(ProjectSettings.globalize_path(pref_path))
	cases.append({"case":"build_preferences_are_separate"})
	volume.detach()
	volume.free()
	layer.free()
	print("AUDIT_JSON="+JSON.stringify({"engine":Engine.get_version_info(),"cases":cases,"failed_assertions":failures}))
	quit(0 if failures == 0 else 1)
