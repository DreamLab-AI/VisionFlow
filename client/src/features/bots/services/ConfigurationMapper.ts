import { BotsVisualConfig } from '../types/BotsTypes';


export interface VisualizationConfig {
  colors: {
    agents: BotsVisualConfig['colors'];
    health: {
      excellent: string;  
      good: string;       
      warning: string;    
      critical: string;   
    };
    edges: {
      active: string;
      inactive: string;
      particle: string;
    };
    background: {
      ambientParticles: string;
      glowEffect: string;
    };
  };
  sizes: {
    nodeBaseSize: number;
    nodeMaxSize: number;
    edgeWidth: number;
    glowScale: number;
    labelFontSize: number;
    labelFontSizeHover: number;
  };
  animation: {
    pulseSpeed: number;
    pulseAmplitude: number;
    particleSpeed: number;
    particleCount: number;
    edgeActivityThreshold: number; 
    staleEdgeThreshold: number;    
  };
  physics: BotsVisualConfig['physics'];
  camera: {
    defaultPosition: [number, number, number];
    defaultTarget: [number, number, number];
    fov: number;
    near: number;
    far: number;
  };
  lighting: {
    ambientIntensity: number;
    ambientColor: string;
    directionalIntensity: number;
    directionalColor: string;
    directionalPosition: [number, number, number];
  };
  rendering: {
    metalness: number;
    roughness: number;
    emissiveIntensity: number;
    nodeOpacity: number;
    edgeOpacity: number;
    glowOpacity: number;
    enableShadows: boolean;
    enablePostProcessing: boolean;
  };
}

export class ConfigurationMapper {
  private static instance: ConfigurationMapper;
  private config: VisualizationConfig;
  private listeners: Map<string, (config: VisualizationConfig) => void> = new Map();

  private constructor() {
    this.config = this.getDefaultConfig();
  }

  static getInstance(): ConfigurationMapper {
    if (!ConfigurationMapper.instance) {
      ConfigurationMapper.instance = new ConfigurationMapper();
    }
    return ConfigurationMapper.instance;
  }

  
  private getDefaultConfig(): VisualizationConfig {
    return {
      colors: {
        agents: {
          
          coder: '#2ECC71',       
          tester: '#27AE60',      
          researcher: '#1ABC9C',  
          reviewer: '#16A085',    
          documenter: '#229954',  
          specialist: '#239B56',  
          
          
          coordinator: '#F1C40F', 
          analyst: '#F39C12',     
          architect: '#E67E22',   
          optimizer: '#D68910',   
          monitor: '#D4AC0D',     
        },
        health: {
          excellent: '#2ECC71',   
          good: '#F1C40F',        
          warning: '#E67E22',     
          critical: '#E74C3C',    
        },
        edges: {
          active: '#FFD700',      
          inactive: '#F1C40F',    
          particle: '#FFD700',    
        },
        background: {
          ambientParticles: '#F1C40F',
          glowEffect: '#F1C40F',
        }
      },
      sizes: {
        nodeBaseSize: 0.5,
        nodeMaxSize: 2.0,
        edgeWidth: 0.05,
        glowScale: 1.8,
        labelFontSize: 0.5,
        labelFontSizeHover: 0.6,
      },
      animation: {
        pulseSpeed: 1.0,
        pulseAmplitude: 0.1,
        particleSpeed: 1.0,
        particleCount: 8,
        edgeActivityThreshold: 5000,  
        staleEdgeThreshold: 30000,    
      },
      physics: {
        springStrength: 0.3,
        linkDistance: 20,
        damping: 0.95,
        nodeRepulsion: 15,
        gravityStrength: 0.1,
        maxVelocity: 0.5,
      },
      camera: {
        defaultPosition: [0, 20, 40],
        defaultTarget: [0, 0, 0],
        fov: 75,
        near: 0.1,
        far: 1000,
      },
      lighting: {
        ambientIntensity: 0.6,
        ambientColor: '#ffffff',
        directionalIntensity: 0.8,
        directionalColor: '#ffffff',
        directionalPosition: [10, 20, 10],
      },
      rendering: {
        metalness: 0.8,
        roughness: 0.2,
        emissiveIntensity: 0.3,
        nodeOpacity: 0.5,
        edgeOpacity: 0.2,
        glowOpacity: 0.1,
        enableShadows: true,
        enablePostProcessing: true,
      }
    };
  }

  
  getConfig(): VisualizationConfig {
    return { ...this.config };
  }

  
  updateConfig(updates: Partial<VisualizationConfig>): void {
    this.config = this.deepMerge(this.config, updates);
    this.notifyListeners();
  }

  
  updatePath(path: string, value: any): void {
    const keys = path.split('.');
    let current: any = this.config;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!current[keys[i]]) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
    this.notifyListeners();
  }

  
  resetToDefault(): void {
    this.config = this.getDefaultConfig();
    this.notifyListeners();
  }

  
  subscribe(id: string, callback: (config: VisualizationConfig) => void): void {
    this.listeners.set(id, callback);
  }

  
  unsubscribe(id: string): void {
    this.listeners.delete(id);
  }

  
  private notifyListeners(): void {
    this.listeners.forEach(callback => {
      callback(this.getConfig());
    });
  }

  
  private deepMerge(target: any, source: any): any {
    const output = { ...target };
    
    if (this.isObject(target) && this.isObject(source)) {
      Object.keys(source).forEach(key => {
        if (this.isObject(source[key])) {
          if (!(key in target)) {
            Object.assign(output, { [key]: source[key] });
          } else {
            output[key] = this.deepMerge(target[key], source[key]);
          }
        } else {
          Object.assign(output, { [key]: source[key] });
        }
      });
    }
    
    return output;
  }

  
  private isObject(item: any): boolean {
    return item && typeof item === 'object' && !Array.isArray(item);
  }

  
  exportConfig(): string {
    return JSON.stringify(this.config, null, 2);
  }

  
  importConfig(json: string): void {
    try {
      const imported = JSON.parse(json);
      this.config = this.deepMerge(this.getDefaultConfig(), imported);
      this.notifyListeners();
    } catch (error) {
      console.error('Failed to import configuration:', error);
      throw new Error('Invalid configuration JSON');
    }
  }

  
  getPresets(): { [key: string]: Partial<VisualizationConfig> } {
    return {
      default: {},
      highPerformance: {
        rendering: {
          enableShadows: false,
          enablePostProcessing: false,
          nodeOpacity: 0.7,
        },
        animation: {
          particleCount: 4,
        }
      },
      darkMode: {
        colors: {
          agents: {
            coder: '#00D9FF',
            tester: '#00B8D4',
            researcher: '#00ACC1',
            reviewer: '#0097A7',
            documenter: '#00838F',
            specialist: '#006064',
            coordinator: '#FFD700',
            analyst: '#FFC107',
            architect: '#FFB300',
            optimizer: '#FFA000',
            monitor: '#FF8F00',
          },
          background: {
            ambientParticles: '#FFD700',
            glowEffect: '#FFD700',
          }
        },
        lighting: {
          ambientIntensity: 0.4,
        }
      },
      presentation: {
        sizes: {
          nodeBaseSize: 0.8,
          nodeMaxSize: 2.5,
          labelFontSize: 0.7,
          labelFontSizeHover: 0.9,
        },
        rendering: {
          emissiveIntensity: 0.5,
          nodeOpacity: 0.8,
        }
      }
    };
  }

  
  applyPreset(presetName: string): void {
    const presets = this.getPresets();
    if (presets[presetName]) {
      this.updateConfig(presets[presetName]);
    }
  }
}

export const configurationMapper = ConfigurationMapper.getInstance();