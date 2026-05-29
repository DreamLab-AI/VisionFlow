using UnityEngine;

namespace VisionFlow.Graph
{
    public sealed class NodeView : MonoBehaviour
    {
        public GraphNode Data { get; private set; }
        public int PoolIndex { get; private set; } = -1;

        [Header("LOD")]
        [SerializeField] private float impostorDistance   = 3f;
        [SerializeField] private float detailDistance     = 1f;
        [SerializeField] private float gazeConeDegrees    = 35f;
        [SerializeField] private float impostorScale      = 0.06f;

        [SerializeField] private Color hoverTint   = new Color(0.35f, 0.85f, 1f);
        [SerializeField] private Color selectTint  = new Color(1f, 0.75f, 0.2f);
        [SerializeField] private float highlightLerpSpeed = 12f;
        [SerializeField] private float selectPulseSpeed   = 6f;
        [SerializeField] private float selectPulseAmount  = 0.08f;

        private Renderer _detailRenderer;
        private Renderer _impostorRenderer;
        private Transform _cameraTransform;
        private Color    _baseColor;
        private Vector3  _baseScale;
        private bool     _hovered;
        private bool     _selected;
        private bool     _culled;
        private Color    _displayColor;
        private LodMode  _lodMode = LodMode.Full;

        private enum LodMode { Full, Impostor }

        private void Awake()
        {
            _detailRenderer = GetComponent<Renderer>();
            EnsureImpostor();
        }

        private void Update()
        {
            if (_culled || Data == null)
                return;

            UpdateLod();
            UpdateHighlight();
        }

        private void OnBecameVisible()
        {
            _culled = false;
            SetRenderersEnabled(true);
        }

        private void OnBecameInvisible()
        {
            _culled = true;
            SetRenderersEnabled(false);
        }

        public void SetCamera(Transform cameraTransform) => _cameraTransform = cameraTransform;

        public void SetPoolIndex(int index) => PoolIndex = index;

        public void SetData(GraphNode data, Color color)
        {
            Data = data;
            if (data == null)
            {
                gameObject.SetActive(false);
                return;
            }

            _baseColor = color;
            _displayColor = color;

            float s = Mathf.Lerp(0.02f, 0.07f, Mathf.Clamp01(data.PageRank * 5f));
            _baseScale = Vector3.one * s;
            transform.localScale = _baseScale;
            ApplyColor(color);
        }

        public void SetHighlight(bool highlighted)
        {
            _hovered = highlighted;
            if (!highlighted && !_selected)
                _displayColor = _baseColor;
        }

        public void SetSelected(bool selected)
        {
            _selected = selected;
            if (!selected && !_hovered)
                _displayColor = _baseColor;
        }

        public void SetFeedHighlight(bool highlighted)
        {
            if (highlighted)
                SetSelected(true);
            else if (!_hovered)
                SetSelected(false);
        }

        private void UpdateLod()
        {
            if (_cameraTransform == null)
            {
                SetLodMode(LodMode.Full);
                return;
            }

            Vector3 toCamera = _cameraTransform.position - transform.position;
            float dist = toCamera.magnitude;

            if (dist > impostorDistance)
            {
                SetLodMode(LodMode.Impostor);
                return;
            }

            if (dist <= detailDistance && IsInGazeCone(toCamera.normalized))
            {
                SetLodMode(LodMode.Full);
                return;
            }

            SetLodMode(LodMode.Impostor);
        }

        private bool IsInGazeCone(Vector3 toNodeNormalized)
        {
            float angle = Vector3.Angle(_cameraTransform.forward, toNodeNormalized);
            return angle <= gazeConeDegrees * 0.5f;
        }

        private void SetLodMode(LodMode mode)
        {
            if (_lodMode == mode) return;
            _lodMode = mode;

            if (_detailRenderer != null)
                _detailRenderer.enabled = mode == LodMode.Full;
            if (_impostorRenderer != null)
            {
                _impostorRenderer.enabled = mode == LodMode.Impostor;
                if (mode == LodMode.Impostor)
                    transform.localScale = Vector3.one * impostorScale;
            }
        }

        private void UpdateHighlight()
        {
            Color target = _baseColor;
            if (_selected)
                target = Color.Lerp(selectTint, _baseColor, 0.35f);
            else if (_hovered)
                target = Color.Lerp(hoverTint, _baseColor, 0.25f);

            _displayColor = Color.Lerp(_displayColor, target, Time.deltaTime * highlightLerpSpeed);
            ApplyColor(_displayColor);

            var scale = _lodMode == LodMode.Full ? _baseScale : Vector3.one * impostorScale;
            if (_selected && _lodMode == LodMode.Full)
            {
                float pulse = 1f + Mathf.Sin(Time.time * selectPulseSpeed) * selectPulseAmount;
                scale *= pulse;
            }

            transform.localScale = Vector3.Lerp(transform.localScale, scale, Time.deltaTime * highlightLerpSpeed);

            if (_lodMode == LodMode.Impostor && _impostorRenderer != null && _cameraTransform != null)
            {
                var look = _cameraTransform.position - transform.position;
                if (look.sqrMagnitude > 1e-6f)
                    transform.rotation = Quaternion.LookRotation(-look.normalized, Vector3.up);
            }
        }

        private void EnsureImpostor()
        {
            var existing = transform.Find("Impostor");
            if (existing != null)
            {
                _impostorRenderer = existing.GetComponent<Renderer>();
                return;
            }

            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quad.name = "Impostor";
            quad.transform.SetParent(transform, false);
            quad.transform.localPosition = Vector3.zero;
            quad.transform.localRotation = Quaternion.identity;
            quad.transform.localScale    = Vector3.one * 0.12f;

            var col = quad.GetComponent<Collider>();
            if (col != null) Destroy(col);

            _impostorRenderer = quad.GetComponent<Renderer>();
            if (_detailRenderer != null && _impostorRenderer != null)
                _impostorRenderer.sharedMaterial = _detailRenderer.sharedMaterial;

            _impostorRenderer.enabled = false;
        }

        private void SetRenderersEnabled(bool enabled)
        {
            if (!enabled)
            {
                if (_detailRenderer != null) _detailRenderer.enabled = false;
                if (_impostorRenderer != null) _impostorRenderer.enabled = false;
                return;
            }

            SetLodMode(_lodMode);
        }

        private void ApplyColor(Color color)
        {
            if (_detailRenderer != null)
                _detailRenderer.material.color = color;
            if (_impostorRenderer != null)
                _impostorRenderer.material.color = color;
        }
    }
}
