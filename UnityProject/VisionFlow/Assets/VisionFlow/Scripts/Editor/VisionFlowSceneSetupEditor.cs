using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using UnityEngine.XR.Interaction.Toolkit.UI;
using Unity.XR.CoreUtils;
using TMPro;
using VisionFlow;
using VisionFlow.Graph;
using VisionFlow.Interaction;
using VisionFlow.UI;

namespace VisionFlow.Editor
{
    /// <summary>
    /// Menu items to scaffold GraphRoot, World Space inspector UI, node prefab, and VisionFlowManager wiring.
    /// </summary>
    public static class VisionFlowSceneSetupEditor
    {
        private const string NodePrefabPath = "Assets/VisionFlow/Prefabs/Nodes/NodeSphere.prefab";
        private const string EdgeMaterialPath = "Assets/VisionFlow/Materials/GraphEdge.mat";

        [MenuItem("GameObject/VisionFlow/Graph Root", false, 10)]
        public static void CreateGraphRootMenu() => CreateGraphRoot();

        [MenuItem("GameObject/VisionFlow/World Space UI", false, 11)]
        public static void CreateWorldSpaceUiMenu() => CreateWorldSpaceUi();

        [MenuItem("GameObject/VisionFlow/Governance UI", false, 12)]
        public static void CreateGovernanceUiMenu() => CreateGovernanceUi();

        [MenuItem("GameObject/VisionFlow/Event Feed UI", false, 13)]
        public static void CreateEventFeedUiMenu() => CreateEventFeedUi();

        [MenuItem("GameObject/VisionFlow/Bootstrap (Manager + Graph + UI)", false, 14)]
        public static void CreateBootstrapMenu() => CreateBootstrap();

        [MenuItem("GameObject/VisionFlow/Wire Manager To Scene Objects", false, 15)]
        public static void WireManagerMenu() => WireManager(FindOrCreateManager());

        public static GameObject CreateGraphRoot(Transform parent = null)
        {
            EnsureFolders();

            var root = new GameObject("GraphRoot");
            Undo.RegisterCreatedObjectUndo(root, "Create Graph Root");
            if (parent != null)
                root.transform.SetParent(parent, false);

            var viz = root.AddComponent<GraphVisualizer>();
            var rb = root.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity  = false;
            root.AddComponent<GraphRootGrab>();

            var nodePrefab = AssetDatabase.LoadAssetAtPath<GameObject>(NodePrefabPath);
            if (nodePrefab == null)
                nodePrefab = CreateNodePrefabAsset();

            var edgeMat = AssetDatabase.LoadAssetAtPath<Material>(EdgeMaterialPath);
            if (edgeMat == null)
                edgeMat = CreateEdgeMaterialAsset();

            var so = new SerializedObject(viz);
            so.FindProperty("nodePrefab").objectReferenceValue = nodePrefab;
            so.FindProperty("edgeMaterial").objectReferenceValue = edgeMat;
            so.ApplyModifiedPropertiesWithoutUndo();

            Selection.activeGameObject = root;
            return root;
        }

        public static GameObject CreateWorldSpaceUi(Transform parent = null)
        {
            var cam = FindXrCameraTransform();

            var canvasGo = new GameObject("WorldSpaceUI");
            Undo.RegisterCreatedObjectUndo(canvasGo, "Create World Space UI");
            if (parent != null)
                canvasGo.transform.SetParent(parent, false);

            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            var rect = canvasGo.GetComponent<RectTransform>();
            rect.sizeDelta = new Vector2(420f, 520f);
            rect.localScale = Vector3.one * 0.001f;
            canvasGo.transform.localPosition = new Vector3(0f, 1.4f, 0.6f);

            var panel = CreateUiChild<RectTransform>(canvasGo.transform, "Panel");
            var panelRect = panel.GetComponent<RectTransform>();
            StretchFull(panelRect);
            var panelImage = panel.gameObject.AddComponent<Image>();
            panelImage.color = new Color(0.05f, 0.08f, 0.12f, 0.92f);

            var title    = CreateTmpLabel(panel.transform, "LabelTitle",    "Node",       new Vector2(0, -24),  24, FontStyles.Bold);
            var type     = CreateTmpLabel(panel.transform, "LabelType",     "Type: —",    new Vector2(0, -64),  18);
            var did      = CreateTmpLabel(panel.transform, "LabelDid",      "DID: …",     new Vector2(0, -96),  16);
            var prov     = CreateTmpLabel(panel.transform, "LabelProvenance","Bead: none", new Vector2(0, -128), 16);
            var pr       = CreateTmpLabel(panel.transform, "LabelPageRank", "PageRank: —", new Vector2(0, -160), 16);
            var comm     = CreateTmpLabel(panel.transform, "LabelCommunity","Community: 0", new Vector2(0, -192), 16);
            var btnClose = CreateButton(panel.transform, "BtnClose", "Close", new Vector2(-90, -240));
            var btnBead  = CreateButton(panel.transform, "BtnWriteBead", "Write bead", new Vector2(90, -240));

            var inspector = canvasGo.AddComponent<NodeInspectorUI>();
            var so = new SerializedObject(inspector);
            so.FindProperty("labelTitle").objectReferenceValue      = title;
            so.FindProperty("labelType").objectReferenceValue       = type;
            so.FindProperty("labelDid").objectReferenceValue       = did;
            so.FindProperty("labelProvenance").objectReferenceValue = prov;
            so.FindProperty("labelPageRank").objectReferenceValue  = pr;
            so.FindProperty("labelCommunity").objectReferenceValue = comm;
            so.FindProperty("btnClose").objectReferenceValue       = btnClose;
            so.FindProperty("btnWriteBead").objectReferenceValue   = btnBead;
            if (cam != null)
                so.FindProperty("xrCameraRig").objectReferenceValue = cam;
            so.ApplyModifiedPropertiesWithoutUndo();

            if (cam == null)
                Debug.LogWarning("[VisionFlow] No XR/Main Camera found — assign xrCameraRig on WorldSpaceUI after adding XR Origin.");

            Selection.activeGameObject = canvasGo;
            return canvasGo;
        }

        public static GameObject CreateGovernanceUi(Transform parent = null)
        {
            var cam = FindXrCameraTransform();

            var canvasGo = new GameObject("GovernanceUI");
            Undo.RegisterCreatedObjectUndo(canvasGo, "Create Governance UI");
            if (parent != null)
                canvasGo.transform.SetParent(parent, false);

            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            var rect = canvasGo.GetComponent<RectTransform>();
            rect.sizeDelta = new Vector2(460f, 560f);
            rect.localScale = Vector3.one * 0.001f;
            canvasGo.transform.localPosition = new Vector3(0.35f, 1.4f, 0.65f);

            var badgeGo = CreateUiChild<RectTransform>(canvasGo.transform, "QueueBadge");
            var badgeRt = badgeGo.GetComponent<RectTransform>();
            badgeRt.anchorMin = badgeRt.anchorMax = new Vector2(1f, 1f);
            badgeRt.pivot = new Vector2(1f, 1f);
            badgeRt.anchoredPosition = new Vector2(-12f, -12f);
            badgeRt.sizeDelta = new Vector2(48f, 48f);
            var badgeBg = badgeGo.gameObject.AddComponent<Image>();
            badgeBg.color = new Color(0.9f, 0.25f, 0.2f, 0.95f);
            var badgeLabel = CreateTmpLabel(badgeGo.transform, "BadgeLabel", "0", Vector2.zero, 22, FontStyles.Bold);
            badgeLabel.GetComponent<RectTransform>().sizeDelta = new Vector2(48f, 48f);
            var badge = badgeGo.gameObject.AddComponent<GovernanceHUDBadge>();
            var badgeSo = new SerializedObject(badge);
            badgeSo.FindProperty("label").objectReferenceValue = badgeLabel;
            badgeSo.FindProperty("badgeRoot").objectReferenceValue = badgeGo.gameObject;
            badgeSo.ApplyModifiedPropertiesWithoutUndo();
            badgeGo.gameObject.SetActive(false);

            var panel = CreateUiChild<RectTransform>(canvasGo.transform, "Panel");
            var panelRect = panel.GetComponent<RectTransform>();
            StretchFull(panelRect);
            var panelImage = panel.gameObject.AddComponent<Image>();
            panelImage.color = new Color(0.08f, 0.06f, 0.12f, 0.94f);

            var title   = CreateTmpLabel(panel.transform, "LabelTitle",   "Governance request", new Vector2(0, -24),  24, FontStyles.Bold);
            var agent   = CreateTmpLabel(panel.transform, "LabelAgent",   "Agent: —",           new Vector2(0, -72),  18);
            var action  = CreateTmpLabel(panel.transform, "LabelAction",  "Action: —",          new Vector2(0, -112), 18);
            var context = CreateTmpLabel(panel.transform, "LabelContext", "Context: —",         new Vector2(0, -180), 16);
            context.GetComponent<RectTransform>().sizeDelta = new Vector2(400f, 80f);
            var queue   = CreateTmpLabel(panel.transform, "LabelQueue",   string.Empty,         new Vector2(0, -260), 16);
            var btnApprove = CreateButton(panel.transform, "BtnApprove", "Approve", new Vector2(-100, -320));
            btnApprove.GetComponent<Image>().color = new Color(0.15f, 0.65f, 0.35f, 1f);
            var btnReject  = CreateButton(panel.transform, "BtnReject",  "Reject",  new Vector2(100, -320));
            btnReject.GetComponent<Image>().color = new Color(0.75f, 0.2f, 0.2f, 1f);

            var reasonGo = CreateUiChild<RectTransform>(panel.transform, "RejectReason");
            var reasonRt = reasonGo.GetComponent<RectTransform>();
            reasonRt.anchorMin = reasonRt.anchorMax = new Vector2(0.5f, 1f);
            reasonRt.pivot = new Vector2(0.5f, 1f);
            reasonRt.anchoredPosition = new Vector2(0, -380);
            reasonRt.sizeDelta = new Vector2(360f, 36f);
            var reasonBg = reasonGo.gameObject.AddComponent<Image>();
            reasonBg.color = new Color(0.12f, 0.12f, 0.14f, 1f);
            var reasonInput = reasonGo.gameObject.AddComponent<TMP_InputField>();
            var reasonText = CreateTmpLabel(reasonGo.transform, "Text", "Reason (optional)", Vector2.zero, 16);
            var reasonTextRt = reasonText.GetComponent<RectTransform>();
            StretchFull(reasonTextRt);
            reasonInput.textViewport = reasonTextRt;
            reasonInput.textComponent = reasonText;
            reasonInput.placeholder = reasonText;
            reasonGo.gameObject.SetActive(false);

            var governance = canvasGo.AddComponent<GovernanceRequestUI>();
            var so = new SerializedObject(governance);
            so.FindProperty("labelTitle").objectReferenceValue   = title;
            so.FindProperty("labelAgent").objectReferenceValue   = agent;
            so.FindProperty("labelAction").objectReferenceValue  = action;
            so.FindProperty("labelContext").objectReferenceValue = context;
            so.FindProperty("labelQueue").objectReferenceValue   = queue;
            so.FindProperty("btnApprove").objectReferenceValue   = btnApprove;
            so.FindProperty("btnReject").objectReferenceValue    = btnReject;
            so.FindProperty("inputRejectReason").objectReferenceValue = reasonInput;
            so.FindProperty("hudBadge").objectReferenceValue     = badge;
            so.FindProperty("reviewPanel").objectReferenceValue  = panel.gameObject;
            if (cam != null)
                so.FindProperty("xrCameraRig").objectReferenceValue = cam;
            so.ApplyModifiedPropertiesWithoutUndo();

            panel.gameObject.SetActive(false);
            Selection.activeGameObject = canvasGo;
            return canvasGo;
        }

        public static GameObject CreateEventFeedUi(Transform parent = null)
        {
            var cam = FindXrCameraTransform();

            var canvasGo = new GameObject("EventFeedUI");
            Undo.RegisterCreatedObjectUndo(canvasGo, "Create Event Feed UI");
            if (parent != null)
                canvasGo.transform.SetParent(parent, false);

            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();
            canvasGo.AddComponent<TrackedDeviceGraphicRaycaster>();

            var rect = canvasGo.GetComponent<RectTransform>();
            rect.sizeDelta = new Vector2(380f, 520f);
            rect.localScale = Vector3.one * 0.001f;
            canvasGo.transform.localPosition = new Vector3(-0.4f, 1.35f, 0.6f);

            var panel = CreateUiChild<RectTransform>(canvasGo.transform, "Panel");
            StretchFull(panel.GetComponent<RectTransform>());
            var panelBg = panel.gameObject.AddComponent<Image>();
            panelBg.color = new Color(0.04f, 0.06f, 0.1f, 0.9f);

            var title = CreateTmpLabel(panel.transform, "LabelTitle", "Nostr feed", new Vector2(0, -20), 22, FontStyles.Bold);
            var empty = CreateTmpLabel(panel.transform, "LabelEmpty", "Waiting for events…", new Vector2(0, -56), 16);
            empty.color = new Color(0.7f, 0.7f, 0.75f, 0.9f);

            var scrollGo = CreateUiChild<RectTransform>(panel.transform, "Scroll");
            var scrollRt = scrollGo.GetComponent<RectTransform>();
            scrollRt.anchorMin = new Vector2(0, 0);
            scrollRt.anchorMax = new Vector2(1, 1);
            scrollRt.offsetMin = new Vector2(12, 12);
            scrollRt.offsetMax = new Vector2(-12, -72);

            var viewport = CreateUiChild<RectTransform>(scrollGo.transform, "Viewport");
            StretchFull(viewport.GetComponent<RectTransform>());
            viewport.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, 0.15f);
            viewport.gameObject.AddComponent<Mask>().showMaskGraphic = false;

            var content = CreateUiChild<RectTransform>(viewport.transform, "Content");
            var contentRt = content.GetComponent<RectTransform>();
            contentRt.anchorMin = new Vector2(0, 1);
            contentRt.anchorMax = new Vector2(1, 1);
            contentRt.pivot = new Vector2(0.5f, 1f);
            contentRt.anchoredPosition = Vector2.zero;
            contentRt.sizeDelta = new Vector2(0, 0);
            var layout = content.gameObject.AddComponent<VerticalLayoutGroup>();
            layout.childAlignment = TextAnchor.UpperCenter;
            layout.spacing = 6;
            layout.childControlWidth = true;
            layout.childControlHeight = true;
            layout.childForceExpandWidth = true;
            layout.childForceExpandHeight = false;
            content.gameObject.AddComponent<ContentSizeFitter>().verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            var scroll = scrollGo.gameObject.AddComponent<ScrollRect>();
            scroll.viewport = viewport.GetComponent<RectTransform>();
            scroll.content = contentRt;
            scroll.horizontal = false;
            scroll.vertical = true;

            var itemRow = CreateFeedItemRow(content.transform, "FeedItemTemplate");
            itemRow.gameObject.SetActive(false);

            var feed = canvasGo.AddComponent<EventFeedUI>();
            var so = new SerializedObject(feed);
            so.FindProperty("contentRoot").objectReferenceValue = contentRt;
            so.FindProperty("itemPrefab").objectReferenceValue = itemRow;
            so.FindProperty("scrollRect").objectReferenceValue = scroll;
            so.FindProperty("labelEmpty").objectReferenceValue = empty;
            if (cam != null)
                so.FindProperty("xrCameraRig").objectReferenceValue = cam;
            so.ApplyModifiedPropertiesWithoutUndo();

            Selection.activeGameObject = canvasGo;
            return canvasGo;
        }

        private static EventFeedItem CreateFeedItemRow(Transform parent, string name)
        {
            var row = CreateUiChild<RectTransform>(parent, name);
            var rowRt = row.GetComponent<RectTransform>();
            rowRt.sizeDelta = new Vector2(340f, 56f);
            var rowLayout = row.gameObject.AddComponent<LayoutElement>();
            rowLayout.preferredHeight = 56;

            var rowBg = row.gameObject.AddComponent<Image>();
            rowBg.color = new Color(0.1f, 0.12f, 0.16f, 0.95f);

            var stripe = CreateUiChild<RectTransform>(row.transform, "Stripe");
            var stripeRt = stripe.GetComponent<RectTransform>();
            stripeRt.anchorMin = new Vector2(0, 0);
            stripeRt.anchorMax = new Vector2(0, 1);
            stripeRt.pivot = new Vector2(0, 0.5f);
            stripeRt.sizeDelta = new Vector2(6f, 0);
            stripeRt.anchoredPosition = Vector2.zero;
            stripe.gameObject.AddComponent<Image>().color = Color.gray;

            var kind = CreateTmpLabel(row.transform, "LabelKind", "Event", new Vector2(16, -8), 14, FontStyles.Bold);
            var kindRt = kind.GetComponent<RectTransform>();
            kindRt.anchorMin = kindRt.anchorMax = new Vector2(0, 1);
            kindRt.pivot = new Vector2(0, 1);
            kindRt.anchoredPosition = new Vector2(16, -8);
            kindRt.sizeDelta = new Vector2(300f, 20f);
            kind.alignment = TextAlignmentOptions.Left;

            var summary = CreateTmpLabel(row.transform, "LabelSummary", "…", new Vector2(16, -30), 13);
            var sumRt = summary.GetComponent<RectTransform>();
            sumRt.anchorMin = sumRt.anchorMax = new Vector2(0, 1);
            sumRt.pivot = new Vector2(0, 1);
            sumRt.anchoredPosition = new Vector2(16, -30);
            sumRt.sizeDelta = new Vector2(300f, 22f);
            summary.alignment = TextAlignmentOptions.Left;

            var btn = CreateButton(row.transform, "BtnSelect", string.Empty, new Vector2(0, 0));
            StretchFull(btn.GetComponent<RectTransform>());
            var btnImg = btn.GetComponent<Image>();
            btnImg.color = new Color(1, 1, 1, 0.02f);

            var item = row.gameObject.AddComponent<EventFeedItem>();
            var itemSo = new SerializedObject(item);
            itemSo.FindProperty("stripe").objectReferenceValue = stripe.GetComponent<Image>();
            itemSo.FindProperty("labelKind").objectReferenceValue = kind;
            itemSo.FindProperty("labelSummary").objectReferenceValue = summary;
            itemSo.FindProperty("btnSelect").objectReferenceValue = btn;
            itemSo.ApplyModifiedPropertiesWithoutUndo();
            return item;
        }

        public static GameObject CreateBootstrap()
        {
            var managerGo = FindOrCreateManager();
            if (managerGo.GetComponent<GraphRaycaster>() == null)
                managerGo.AddComponent<GraphRaycaster>();
            if (managerGo.GetComponent<GestureMapper>() == null)
                managerGo.AddComponent<GestureMapper>();

            var graph = CreateGraphRoot(managerGo.transform);
            var ui    = CreateWorldSpaceUi(managerGo.transform);
            var gov   = CreateGovernanceUi(managerGo.transform);
            var feed  = CreateEventFeedUi(managerGo.transform);
            WireManager(
                managerGo,
                graph.GetComponent<GraphVisualizer>(),
                ui.GetComponent<NodeInspectorUI>(),
                gov.GetComponent<GovernanceRequestUI>(),
                feed.GetComponent<EventFeedUI>());

            var raycaster = managerGo.GetComponent<GraphRaycaster>();
            if (raycaster != null)
                WireGraphRaycaster(raycaster);

            Selection.activeGameObject = managerGo;
            Debug.Log("[VisionFlow] Bootstrap done. NearFar + hand pinch (GestureMapper) select graph nodes.");
            return managerGo;
        }

        private static GameObject FindOrCreateManager()
        {
            var existing = Object.FindAnyObjectByType<VisionFlowManager>();
            if (existing != null)
                return existing.gameObject;

            var go = new GameObject("VisionFlow");
            Undo.RegisterCreatedObjectUndo(go, "Create VisionFlow Manager");
            go.AddComponent<VisionFlowManager>();
            return go;
        }

        private static void WireManager(GameObject managerGo)
        {
            var viz = Object.FindAnyObjectByType<GraphVisualizer>();
            var ui  = Object.FindAnyObjectByType<NodeInspectorUI>();
            var gov = Object.FindAnyObjectByType<GovernanceRequestUI>();
            var feed = Object.FindAnyObjectByType<EventFeedUI>();
            WireManager(managerGo, viz, ui, gov, feed);
        }

        private static void WireManager(
            GameObject managerGo,
            GraphVisualizer viz,
            NodeInspectorUI ui,
            GovernanceRequestUI governance = null,
            EventFeedUI eventFeed = null)
        {
            var manager = managerGo.GetComponent<VisionFlowManager>();
            if (manager == null)
            {
                Debug.LogError("[VisionFlow] VisionFlowManager missing on " + managerGo.name);
                return;
            }

            if (governance == null)
                governance = Object.FindAnyObjectByType<GovernanceRequestUI>();
            if (eventFeed == null)
                eventFeed = Object.FindAnyObjectByType<EventFeedUI>();

            var so = new SerializedObject(manager);
            so.FindProperty("graphVisualizer").objectReferenceValue = viz;
            so.FindProperty("nodeInspectorUI").objectReferenceValue = ui;
            so.FindProperty("governanceRequestUI").objectReferenceValue = governance;
            so.FindProperty("eventFeedUI").objectReferenceValue = eventFeed;
            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(manager);
            Debug.Log("[VisionFlow] VisionFlowManager wired (graph, inspector, governance, event feed).");
        }

        private static GameObject CreateNodePrefabAsset()
        {
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = "NodeSphere";
            var col = sphere.GetComponent<SphereCollider>();
            col.radius = 0.06f;
            sphere.AddComponent<NodeView>();
            sphere.AddComponent<NodeInteractable>();

            var prefab = PrefabUtility.SaveAsPrefabAsset(sphere, NodePrefabPath);
            Object.DestroyImmediate(sphere);
            AssetDatabase.SaveAssets();
            Debug.Log("[VisionFlow] Created node prefab at " + NodePrefabPath);
            return prefab;
        }

        private static Material CreateEdgeMaterialAsset()
        {
            EnsureFolders();
            var shader = Shader.Find("Universal Render Pipeline/Unlit")
                         ?? Shader.Find("Unlit/Color")
                         ?? Shader.Find("Sprites/Default");
            var mat = new Material(shader) { name = "GraphEdge", color = new Color(0.65f, 0.65f, 0.65f, 0.85f) };
            AssetDatabase.CreateAsset(mat, EdgeMaterialPath);
            AssetDatabase.SaveAssets();
            return mat;
        }

        private static void EnsureFolders()
        {
            EnsureFolder("Assets/VisionFlow/Prefabs/Nodes");
            EnsureFolder("Assets/VisionFlow/Materials");
        }

        private static void EnsureFolder(string path)
        {
            if (AssetDatabase.IsValidFolder(path)) return;
            var parts = path.Split('/');
            var built = parts[0];
            for (var i = 1; i < parts.Length; i++)
            {
                var next = built + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(built, parts[i]);
                built = next;
            }
        }

        private static Transform FindXrCameraTransform()
        {
            if (Camera.main != null)
                return Camera.main.transform;

            var main = GameObject.Find("Main Camera");
            if (main != null)
                return main.transform;

            var cam = Object.FindAnyObjectByType<Camera>();
            return cam != null ? cam.transform : null;
        }

        private static T CreateUiChild<T>(Transform parent, string name) where T : Component
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            return go.GetComponent<T>();
        }

        private static void StretchFull(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
        }

        private static TextMeshProUGUI CreateTmpLabel(Transform parent, string name, string text, Vector2 anchoredPos, int fontSize, FontStyles style = FontStyles.Normal)
        {
            var go = new GameObject(name, typeof(RectTransform));
            Undo.RegisterCreatedObjectUndo(go, "Create UI Label");
            go.transform.SetParent(parent, false);

            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = anchoredPos;
            rt.sizeDelta = new Vector2(380f, 32f);

            var tmp = go.AddComponent<TextMeshProUGUI>();
            tmp.text = text;
            tmp.fontSize = fontSize;
            tmp.fontStyle = style;
            tmp.color = Color.white;
            tmp.alignment = TextAlignmentOptions.Center;
            tmp.enableWordWrapping = true;
            return tmp;
        }

        /// <summary>Wire <see cref="GraphRaycaster"/> to NearFarInteractor instances on the XR rig.</summary>
        public static void WireGraphRaycaster(GraphRaycaster raycaster, Transform xrOriginRoot = null)
        {
            if (raycaster == null) return;

            if (xrOriginRoot == null)
            {
                var origin = Object.FindAnyObjectByType<XROrigin>();
                xrOriginRoot = origin != null ? origin.transform : null;
            }

            NearFarInteractor left = null, right = null;
            var all = xrOriginRoot != null
                ? xrOriginRoot.GetComponentsInChildren<NearFarInteractor>(true)
                : Object.FindObjectsByType<NearFarInteractor>(FindObjectsInactive.Include);

            foreach (var nf in all)
            {
                switch (nf.handedness)
                {
                    case InteractorHandedness.Left when left == null:
                        left = nf;
                        break;
                    case InteractorHandedness.Right when right == null:
                        right = nf;
                        break;
                }
            }

            if (left == null || right == null)
            {
                foreach (var nf in all)
                {
                    var n = nf.gameObject.name.ToLowerInvariant();
                    if (left == null && n.Contains("left"))  left  = nf;
                    if (right == null && n.Contains("right")) right = nf;
                }
            }

            if (left == null || right == null)
            {
                if (left == null && all.Length > 0)  left  = all[0];
                if (right == null && all.Length > 1) right = all[1];
            }

            var so = new SerializedObject(raycaster);
            if (xrOriginRoot != null)
            {
                var origin = xrOriginRoot.GetComponent<XROrigin>()
                             ?? xrOriginRoot.GetComponentInParent<XROrigin>();
                so.FindProperty("xrOrigin").objectReferenceValue = origin;
            }

            so.FindProperty("leftNearFar").objectReferenceValue  = left;
            so.FindProperty("rightNearFar").objectReferenceValue = right;
            so.ApplyModifiedProperties();
            EditorUtility.SetDirty(raycaster);

            if (left != null && right != null)
                Debug.Log("[VisionFlow] GraphRaycaster → NearFar left + right.");
            else if (left != null || right != null)
                Debug.LogWarning("[VisionFlow] GraphRaycaster: only one NearFarInteractor wired.");
            else
                Debug.LogError("[VisionFlow] No NearFarInteractor under XROrigin.");
        }

        private static Button CreateButton(Transform parent, string name, string label, Vector2 anchoredPos)
        {
            var go = new GameObject(name, typeof(RectTransform));
            Undo.RegisterCreatedObjectUndo(go, "Create UI Button");
            go.transform.SetParent(parent, false);

            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = anchoredPos;
            rt.sizeDelta = new Vector2(140f, 40f);

            var image = go.AddComponent<Image>();
            image.color = new Color(0.1f, 0.55f, 1f, 1f);

            var btn = go.AddComponent<Button>();
            btn.targetGraphic = image;

            var textGo = new GameObject("Text", typeof(RectTransform));
            textGo.transform.SetParent(go.transform, false);
            var textRt = textGo.GetComponent<RectTransform>();
            StretchFull(textRt);

            var tmp = textGo.AddComponent<TextMeshProUGUI>();
            tmp.text = label;
            tmp.fontSize = 18;
            tmp.color = Color.white;
            tmp.alignment = TextAlignmentOptions.Center;

            return btn;
        }
    }
}
