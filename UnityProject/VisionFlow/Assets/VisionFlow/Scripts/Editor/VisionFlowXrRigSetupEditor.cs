using UnityEditor;
using UnityEngine;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using Unity.XR.CoreUtils;
using VisionFlow.Interaction;

namespace VisionFlow.Editor
{
    public static class VisionFlowXrRigSetupEditor
    {
        private const string XrOriginPrefabPath =
            "Assets/Samples/XR Interaction Toolkit/3.3.0/Starter Assets/Prefabs/XR Origin (XR Rig).prefab";

        [MenuItem("GameObject/VisionFlow/Wire Near-Far Interactors", false, 20)]
        public static void WireNearFarInteractors()
        {
            var origin = Object.FindAnyObjectByType<XROrigin>();
            if (origin == null)
            {
                Debug.LogError("[VisionFlow] No XROrigin in scene.");
                return;
            }

            if (Object.FindAnyObjectByType<XRInteractionManager>() == null)
            {
                Debug.LogError("[VisionFlow] No XRInteractionManager in scene.");
                return;
            }

            var raycaster = Object.FindAnyObjectByType<GraphRaycaster>();
            if (raycaster == null)
            {
                Debug.LogError("[VisionFlow] No GraphRaycaster — run Bootstrap on VisionFlow first.");
                return;
            }

            VisionFlowSceneSetupEditor.WireGraphRaycaster(raycaster, origin.transform);
        }

        [MenuItem("GameObject/VisionFlow/Add XR Origin (XR Rig) From Samples", false, 21)]
        public static void AddXrOriginFromSamples()
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(XrOriginPrefabPath);
            if (prefab == null)
            {
                Debug.LogError($"[VisionFlow] Missing prefab: {XrOriginPrefabPath}");
                return;
            }

            if (Object.FindAnyObjectByType<XROrigin>() != null)
            {
                Debug.LogWarning("[VisionFlow] Scene already has XROrigin.");
                return;
            }

            var instance = PrefabUtility.InstantiatePrefab(prefab) as GameObject;
            if (instance == null) return;

            Undo.RegisterCreatedObjectUndo(instance, "Add XR Origin");
            instance.transform.position = Vector3.zero;
            Selection.activeGameObject = instance;
            Debug.Log("[VisionFlow] XR Origin added (NearFarInteractor on each controller). Run Wire Near-Far Interactors after Bootstrap.");
        }

    }
}
