import axios, { AxiosResponse } from 'axios';
import {
  NeuralAgent,
  SwarmTopology,
  NeuralMemory,
  ConsensusState,
  ResourceMetrics,
  TaskResult,
  CognitivePattern
} from '../types/neural';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

class NeuralAPI {
  private baseURL: string;
  private token?: string;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.setupInterceptors();
  }

  private setupInterceptors() {
    axios.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      return config;
    });

    axios.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        throw error;
      }
    );
  }

  setAuthToken(token: string) {
    this.token = token;
  }

  // Agent Management
  async getAgents(): Promise<NeuralAgent[]> {
    const response: AxiosResponse<NeuralAgent[]> = await axios.get(`${this.baseURL}/api/neural/agents`);
    return response.data;
  }

  async getAgent(agentId: string): Promise<NeuralAgent> {
    const response: AxiosResponse<NeuralAgent> = await axios.get(`${this.baseURL}/api/neural/agents/${agentId}`);
    return response.data;
  }

  async createAgent(agentData: Partial<NeuralAgent>): Promise<NeuralAgent> {
    const response: AxiosResponse<NeuralAgent> = await axios.post(`${this.baseURL}/api/neural/agents`, agentData);
    return response.data;
  }

  async updateAgent(agentId: string, updates: Partial<NeuralAgent>): Promise<NeuralAgent> {
    const response: AxiosResponse<NeuralAgent> = await axios.patch(`${this.baseURL}/api/neural/agents/${agentId}`, updates);
    return response.data;
  }

  async deleteAgent(agentId: string): Promise<void> {
    await axios.delete(`${this.baseURL}/api/neural/agents/${agentId}`);
  }

  async adaptAgent(agentId: string, feedback: string, performanceScore: number): Promise<NeuralAgent> {
    const response: AxiosResponse<NeuralAgent> = await axios.post(`${this.baseURL}/api/neural/agents/${agentId}/adapt`, {
      feedback,
      performanceScore
    });
    return response.data;
  }

  // Swarm Management
  async initializeSwarm(topology: string, maxAgents: number = 8): Promise<SwarmTopology> {
    const response: AxiosResponse<SwarmTopology> = await axios.post(`${this.baseURL}/api/neural/swarm/init`, {
      topology,
      maxAgents
    });
    return response.data;
  }

  async getSwarmStatus(): Promise<SwarmTopology> {
    const response: AxiosResponse<SwarmTopology> = await axios.get(`${this.baseURL}/api/neural/swarm/status`);
    return response.data;
  }

  async scaleSwarm(targetAgents: number): Promise<SwarmTopology> {
    const response: AxiosResponse<SwarmTopology> = await axios.post(`${this.baseURL}/api/neural/swarm/scale`, {
      targetAgents
    });
    return response.data;
  }

  async destroySwarm(): Promise<void> {
    await axios.delete(`${this.baseURL}/api/neural/swarm`);
  }

  // Memory Management
  async getMemories(type?: string): Promise<NeuralMemory[]> {
    const params = type ? { type } : {};
    const response: AxiosResponse<NeuralMemory[]> = await axios.get(`${this.baseURL}/api/neural/memory`, { params });
    return response.data;
  }

  async storeMemory(memory: Omit<NeuralMemory, 'id' | 'created' | 'lastAccessed' | 'accessCount'>): Promise<NeuralMemory> {
    const response: AxiosResponse<NeuralMemory> = await axios.post(`${this.baseURL}/api/neural/memory`, memory);
    return response.data;
  }

  async retrieveMemory(memoryId: string): Promise<NeuralMemory> {
    const response: AxiosResponse<NeuralMemory> = await axios.get(`${this.baseURL}/api/neural/memory/${memoryId}`);
    return response.data;
  }

  async shareKnowledge(sourceAgentId: string, targetAgentIds: string[], knowledgeDomain: string, content: any): Promise<void> {
    await axios.post(`${this.baseURL}/api/neural/knowledge/share`, {
      sourceAgentId,
      targetAgentIds,
      knowledgeDomain,
      content
    });
  }

  // Consensus Management
  async getConsensusState(): Promise<ConsensusState> {
    const response: AxiosResponse<ConsensusState> = await axios.get(`${this.baseURL}/api/neural/consensus`);
    return response.data;
  }

  async createProposal(content: any): Promise<string> {
    const response: AxiosResponse<{ proposalId: string }> = await axios.post(`${this.baseURL}/api/neural/consensus/proposal`, {
      content
    });
    return response.data.proposalId;
  }

  async voteOnProposal(proposalId: string, vote: 'accept' | 'reject' | 'abstain'): Promise<void> {
    await axios.post(`${this.baseURL}/api/neural/consensus/vote`, {
      proposalId,
      vote
    });
  }

  // Resource Monitoring
  async getResourceMetrics(timeRange?: string): Promise<ResourceMetrics[]> {
    const params = timeRange ? { timeRange } : {};
    const response: AxiosResponse<ResourceMetrics[]> = await axios.get(`${this.baseURL}/api/neural/metrics`, { params });
    return response.data;
  }

  async getPerformanceMetrics(category?: string): Promise<any> {
    const params = category ? { category } : {};
    const response: AxiosResponse<any> = await axios.get(`${this.baseURL}/api/neural/performance`, { params });
    return response.data;
  }

  // Task Management
  async orchestrateTask(task: string, strategy?: string, maxAgents?: number): Promise<string> {
    const response: AxiosResponse<{ taskId: string }> = await axios.post(`${this.baseURL}/api/neural/tasks/orchestrate`, {
      task,
      strategy,
      maxAgents
    });
    return response.data.taskId;
  }

  async getTaskStatus(taskId?: string): Promise<TaskResult[]> {
    const url = taskId ? `${this.baseURL}/api/neural/tasks/${taskId}` : `${this.baseURL}/api/neural/tasks`;
    const response: AxiosResponse<TaskResult[]> = await axios.get(url);
    return response.data;
  }

  async getTaskResults(taskId: string, format?: string): Promise<any> {
    const params = format ? { format } : {};
    const response: AxiosResponse<any> = await axios.get(`${this.baseURL}/api/neural/tasks/${taskId}/results`, { params });
    return response.data;
  }

  // Cognitive Patterns
  async getCognitivePatterns(pattern?: string): Promise<CognitivePattern[]> {
    const params = pattern ? { pattern } : {};
    const response: AxiosResponse<CognitivePattern[]> = await axios.get(`${this.baseURL}/api/neural/patterns`, { params });
    return response.data;
  }

  async changeCognitivePattern(agentId: string, pattern: string): Promise<void> {
    await axios.post(`${this.baseURL}/api/neural/agents/${agentId}/pattern`, {
      pattern
    });
  }

  async enableMetaLearning(sourceDomain: string, targetDomain: string, transferMode?: string): Promise<void> {
    await axios.post(`${this.baseURL}/api/neural/meta-learning`, {
      sourceDomain,
      targetDomain,
      transferMode
    });
  }

  // Training & Learning
  async trainAgent(agentId: string, iterations: number = 10): Promise<void> {
    await axios.post(`${this.baseURL}/api/neural/agents/${agentId}/train`, {
      iterations
    });
  }

  async getLearningStatus(agentId?: string): Promise<any> {
    const url = agentId ?
      `${this.baseURL}/api/neural/learning/${agentId}` :
      `${this.baseURL}/api/neural/learning`;
    const response: AxiosResponse<any> = await axios.get(url);
    return response.data;
  }

  // Benchmarking
  async runBenchmark(type?: string, iterations: number = 10): Promise<any> {
    const response: AxiosResponse<any> = await axios.post(`${this.baseURL}/api/neural/benchmark`, {
      type,
      iterations
    });
    return response.data;
  }

  // WebSocket Health Check
  async checkWebSocketHealth(): Promise<boolean> {
    try {
      const response: AxiosResponse<{ status: string }> = await axios.get(`${this.baseURL}/api/neural/ws/health`);
      return response.data.status === 'healthy';
    } catch {
      return false;
    }
  }
}

export const neuralAPI = new NeuralAPI();
export default neuralAPI;