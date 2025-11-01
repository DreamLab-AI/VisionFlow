import React, { forwardRef, useMemo } from 'react';
import { Uniform } from 'three';
import { Effect } from 'postprocessing';
import * as THREE from 'three';

const fragmentShader = `
  uniform sampler2D sceneTex;
  uniform sampler2D depthTex;
  uniform vec3 glowColor;
  uniform float intensity;
  uniform float radius;
  uniform float threshold;
  uniform float diffuseStrength;
  uniform float atmosphericDensity;
  uniform float volumetricIntensity;
  uniform mat4 projectionMatrix;
  uniform mat4 viewMatrix;
  uniform vec2 resolution;

  float readDepth(vec2 uv) {
    return texture2D(depthTex, uv).r;
  }

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    vec4 baseColor = texture2D(sceneTex, uv);
    float depth = readDepth(uv);

    vec3 viewDir = vec3((uv - 0.5) * 2.0, 1.0);
    
    vec4 worldPos = inverse(projectionMatrix * viewMatrix) * vec4(viewDir * depth, 1.0);
    worldPos /= worldPos.w;

    float glow = 0.0;
    float stepSize = radius / 16.0;

    for(int i = 0; i < 16; i++) {
      vec2 offset = vec2(float(i) / 16.0, 0.0);
      vec4 sampleColor = texture2D(sceneTex, uv + offset);
      float brightness = dot(sampleColor.rgb, vec3(0.299, 0.587, 0.114));
      if(brightness > threshold) {
        glow += (1.0 - float(i)/16.0) * brightness;
      }
    }

    vec3 finalGlow = glowColor * glow * intensity * diffuseStrength;
    
    outputColor = vec4(baseColor.rgb + finalGlow, baseColor.a);
  }
`;

class AtmosphericGlowEffect extends Effect {
  constructor({
    glowColor = new THREE.Color(0x00ffff),
    intensity = 1.0,
    radius = 1.0,
    threshold = 0.5,
    diffuseStrength = 1.0,
    atmosphericDensity = 1.0,
    volumetricIntensity = 1.0,
    camera,
    resolution,
  } = {}) {
    
    let colorUniform;
    if (typeof glowColor === 'string') {
      const color = new THREE.Color(glowColor);
      colorUniform = new THREE.Vector3(color.r, color.g, color.b);
    } else if (glowColor instanceof THREE.Color) {
      colorUniform = new THREE.Vector3(glowColor.r, glowColor.g, glowColor.b);
    } else if (Array.isArray(glowColor)) {
      colorUniform = new THREE.Vector3(glowColor[0], glowColor[1], glowColor[2]);
    } else {
      
      colorUniform = glowColor;
    }
    
    super('AtmosphericGlowEffect', fragmentShader, {
      uniforms: new Map([
        ['sceneTex', new Uniform(null)],
        ['depthTex', new Uniform(null)],
        ['glowColor', new Uniform(colorUniform)],
        ['intensity', new Uniform(intensity)],
        ['radius', new Uniform(radius)],
        ['threshold', new Uniform(threshold)],
        ['diffuseStrength', new Uniform(diffuseStrength)],
        ['atmosphericDensity', new Uniform(atmosphericDensity)],
        ['volumetricIntensity', new Uniform(volumetricIntensity)],
        ['projectionMatrix', new Uniform(camera?.projectionMatrix)],
        ['viewMatrix', new Uniform(camera?.matrixWorldInverse)],
        ['resolution', new Uniform(resolution)],
      ]),
    });
  }
}

export const AtmosphericGlow = forwardRef(({ ...props }, ref) => {
  const effect = useMemo(() => new AtmosphericGlowEffect(props), [props]);
  return <primitive ref={ref} object={effect} dispose={null} />;
});