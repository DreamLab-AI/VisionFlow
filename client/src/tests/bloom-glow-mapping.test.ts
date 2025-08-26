// Test file for bloom/glow field mapping functionality
import { 
  transformBloomToGlow, 
  transformGlowToBloom, 
  normalizeBloomGlowSettings 
} from '../utils/caseConversion';

describe('Bloom/Glow Field Mapping', () => {
  // Test data with bloom settings
  const sampleBloomSettings = {
    visualisation: {
      bloom: {
        enabled: true,
        strength: 2.5,
        nodeBloomStrength: 3.0,
        edgeBloomStrength: 3.5,
        environmentBloomStrength: 2.8,
        radius: 0.85,
        threshold: 0.15
      },
      rendering: {
        backgroundColor: '#0a0e1a'
      }
    }
  };

  // Test data with glow settings
  const sampleGlowSettings = {
    visualisation: {
      glow: {
        enabled: true,
        intensity: 2.5,
        nodeGlowStrength: 3.0,
        edgeGlowStrength: 3.5,
        environmentGlowStrength: 2.8,
        radius: 0.85,
        threshold: 0.15,
        diffuseStrength: 1.5,
        atmosphericDensity: 0.8
      },
      rendering: {
        backgroundColor: '#0a0e1a'
      }
    }
  };

  describe('transformBloomToGlow', () => {
    it('should transform bloom settings to glow settings', () => {
      const result = transformBloomToGlow(sampleBloomSettings);
      
      expect(result.visualisation.glow).toBeDefined();
      expect(result.visualisation.glow.enabled).toBe(true);
      expect(result.visualisation.glow.intensity).toBe(2.5); // bloom.strength -> glow.intensity
      expect(result.visualisation.glow.nodeGlowStrength).toBe(3.0);
      expect(result.visualisation.glow.edgeGlowStrength).toBe(3.5);
      expect(result.visualisation.glow.environmentGlowStrength).toBe(2.8);
    });

    it('should preserve non-bloom settings', () => {
      const result = transformBloomToGlow(sampleBloomSettings);
      
      expect(result.visualisation.rendering.backgroundColor).toBe('#0a0e1a');
    });

    it('should handle empty or null input', () => {
      expect(transformBloomToGlow(null)).toBe(null);
      expect(transformBloomToGlow(undefined)).toBe(undefined);
      expect(transformBloomToGlow({})).toEqual({});
    });
  });

  describe('transformGlowToBloom', () => {
    it('should transform glow settings to bloom settings for client compatibility', () => {
      const result = transformGlowToBloom(sampleGlowSettings);
      
      expect(result.visualisation.bloom).toBeDefined();
      expect(result.visualisation.bloom.enabled).toBe(true);
      expect(result.visualisation.bloom.strength).toBe(2.5); // glow.intensity -> bloom.strength
      expect(result.visualisation.bloom.nodeBloomStrength).toBe(3.0);
      expect(result.visualisation.bloom.edgeBloomStrength).toBe(3.5);
      expect(result.visualisation.bloom.environmentBloomStrength).toBe(2.8);
    });

    it('should preserve original glow settings', () => {
      const result = transformGlowToBloom(sampleGlowSettings);
      
      expect(result.visualisation.glow).toBeDefined();
      expect(result.visualisation.glow.diffuseStrength).toBe(1.5);
      expect(result.visualisation.glow.atmosphericDensity).toBe(0.8);
    });
  });

  describe('normalizeBloomGlowSettings', () => {
    it('should normalize settings for server (toServer)', () => {
      const result = normalizeBloomGlowSettings(sampleBloomSettings, 'toServer');
      
      expect(result.visualisation.glow).toBeDefined();
      expect(result.visualisation.glow.nodeGlowStrength).toBe(3.0);
    });

    it('should normalize settings for client (toClient)', () => {
      const result = normalizeBloomGlowSettings(sampleGlowSettings, 'toClient');
      
      expect(result.visualisation.bloom).toBeDefined();
      expect(result.visualisation.glow).toBeDefined(); // Both should exist
    });

    it('should handle settings with both bloom and glow', () => {
      const mixedSettings = {
        visualisation: {
          bloom: {
            enabled: false,
            strength: 1.5
          },
          glow: {
            enabled: true,
            intensity: 2.5
          }
        }
      };

      const clientResult = normalizeBloomGlowSettings(mixedSettings, 'toClient');
      const serverResult = normalizeBloomGlowSettings(mixedSettings, 'toServer');

      // Client should have both with compatibility mapping
      expect(clientResult.visualisation.bloom).toBeDefined();
      expect(clientResult.visualisation.glow).toBeDefined();

      // Server should prioritize glow
      expect(serverResult.visualisation.glow).toBeDefined();
    });
  });

  describe('Real-world scenarios', () => {
    it('should handle settings from server (glow) for client use', () => {
      const serverResponse = {
        visualisation: {
          glow: {
            enabled: true,
            intensity: 2.0,
            nodeGlowStrength: 3.0,
            edgeGlowStrength: 3.5,
            environmentGlowStrength: 2.8
          }
        }
      };

      const clientSettings = normalizeBloomGlowSettings(serverResponse, 'toClient');

      // Client should have both glow (original) and bloom (compatibility)
      expect(clientSettings.visualisation.glow.enabled).toBe(true);
      expect(clientSettings.visualisation.bloom.enabled).toBe(true);
      expect(clientSettings.visualisation.bloom.strength).toBe(2.0);
      expect(clientSettings.visualisation.bloom.nodeBloomStrength).toBe(3.0);
    });

    it('should handle client settings (bloom) for server submission', () => {
      const clientSettings = {
        visualisation: {
          bloom: {
            enabled: true,
            strength: 2.0,
            nodeBloomStrength: 3.0,
            edgeBloomStrength: 3.5,
            environmentBloomStrength: 2.8
          }
        }
      };

      const serverSettings = normalizeBloomGlowSettings(clientSettings, 'toServer');

      // Server should receive glow settings
      expect(serverSettings.visualisation.glow.enabled).toBe(true);
      expect(serverSettings.visualisation.glow.intensity).toBe(2.0);
      expect(serverSettings.visualisation.glow.nodeGlowStrength).toBe(3.0);
      expect(serverSettings.visualisation.glow.edgeGlowStrength).toBe(3.5);
      expect(serverSettings.visualisation.glow.environmentGlowStrength).toBe(2.8);
    });
  });
});

// Export test utilities for other components to use
export const bloomGlowTestUtils = {
  sampleBloomSettings,
  sampleGlowSettings
};