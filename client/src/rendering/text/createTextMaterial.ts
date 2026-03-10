import * as THREE from 'three';

export interface TextMaterialResult {
  material: THREE.ShaderMaterial;
  uniforms: {
    uAtlas: { value: THREE.Texture };
    uCamRight: { value: THREE.Vector3 };
    uCamUp: { value: THREE.Vector3 };
  };
}

const vertexShader = /* glsl */ `
  // Per-instance attributes
  attribute vec3 aLabelPos;    // node world position + Y offset
  attribute vec2 aLocalOffset; // glyph offset within label
  attribute vec2 aScale;       // glyph quad dimensions
  attribute vec4 aUVRect;      // atlas UV rectangle (u, v, w, h)
  attribute vec3 aColor;       // per-glyph color
  attribute float aOpacity;    // distance fade opacity

  // Camera basis vectors for billboarding
  uniform vec3 uCamRight;
  uniform vec3 uCamUp;

  varying vec2 vUV;
  varying vec3 vColor;
  varying float vOpacity;

  void main() {
    // Compute UV from base quad position (0..1 range)
    vUV = vec2(
      aUVRect.x + position.x * aUVRect.z,
      aUVRect.y + (1.0 - position.y) * aUVRect.w
    );
    vColor = aColor;
    vOpacity = aOpacity;

    // Billboard: orient quad to face camera using camera basis vectors
    vec3 localOffset = uCamRight * (aLocalOffset.x + position.x * aScale.x)
                     + uCamUp * (aLocalOffset.y + position.y * aScale.y);

    vec3 worldPos = aLabelPos + localOffset;

    gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  uniform sampler2D uAtlas;

  varying vec2 vUV;
  varying vec3 vColor;
  varying float vOpacity;

  void main() {
    vec4 texSample = texture2D(uAtlas, vUV);
    float alpha = texSample.r * vOpacity;
    if (alpha < 0.01) discard;
    gl_FragColor = vec4(vColor * texSample.r, alpha);
  }
`;

let _cached: TextMaterialResult | null = null;

export function createTextMaterial(atlas: THREE.Texture): TextMaterialResult {
  if (_cached) return _cached;

  const uniforms = {
    uAtlas: { value: atlas },
    uCamRight: { value: new THREE.Vector3(1, 0, 0) },
    uCamUp: { value: new THREE.Vector3(0, 1, 0) },
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader,
    fragmentShader,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending,
  });

  _cached = { material, uniforms };
  return _cached;
}
