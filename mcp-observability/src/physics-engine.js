import { createLogger } from './logger.js';

const logger = createLogger('PhysicsEngine');

export class PhysicsEngine {
  constructor() {
    this.config = {
      springStrength: 0.1,
      linkDistance: 8.0,
      damping: 0.95,
      nodeRepulsion: 500.0,
      gravityStrength: 0.02,
      maxVelocity: 2.0,
      
      // Hive-mind specific forces
      queenGravity: 0.05,
      swarmCohesion: 0.08,
      hierarchicalForce: 0.03,
      
      // Message flow forces
      messageAttraction: 0.15,
      communicationDecay: 0.98,
    };
    
    this.frameCount = 0;
    this.lastUpdateTime = Date.now();
  }
  
  // Update physics configuration
  updateConfig(newConfig) {
    Object.assign(this.config, newConfig);
    logger.info('Updated physics config:', newConfig);
  }
  
  // Main physics update loop
  update(deltaTime, agents) {
    this.frameCount++;
    const now = Date.now();
    const actualDelta = (now - this.lastUpdateTime) / 1000;
    this.lastUpdateTime = now;
    
    // Update each agent's position based on forces
    agents.forEach(agent => {
      this.updateAgentPhysics(agent, agents, deltaTime);
    });
    
    // Log performance metrics every 60 frames
    if (this.frameCount % 60 === 0) {
      logger.debug(`Physics update: ${agents.length} agents, ${(1/actualDelta).toFixed(1)} FPS`);
    }
  }
  
  // Update individual agent physics
  updateAgentPhysics(agent, allAgents, deltaTime) {
    const force = { x: 0, y: 0, z: 0 };
    
    // Apply forces from other agents
    allAgents.forEach(otherAgent => {
      if (otherAgent.id === agent.id) return;
      
      const forceComponent = this.calculateInteractionForce(agent, otherAgent);
      force.x += forceComponent.x;
      force.y += forceComponent.y;
      force.z += forceComponent.z;
    });
    
    // Apply special forces based on agent type
    if (agent.type !== 'queen' && agent.type !== 'coordinator') {
      const hierarchicalForce = this.calculateHierarchicalForce(agent, allAgents);
      force.x += hierarchicalForce.x;
      force.y += hierarchicalForce.y;
      force.z += hierarchicalForce.z;
    }
    
    // Apply central gravity to prevent drift
    const gravityForce = this.calculateGravityForce(agent.position);
    force.x += gravityForce.x;
    force.y += gravityForce.y;
    force.z += gravityForce.z;
    
    // Apply swarm cohesion force
    const cohesionForce = this.calculateSwarmCohesion(agent, allAgents);
    force.x += cohesionForce.x;
    force.y += cohesionForce.y;
    force.z += cohesionForce.z;
    
    // Update velocity based on force
    agent.velocity.x += force.x * deltaTime;
    agent.velocity.y += force.y * deltaTime;
    agent.velocity.z += force.z * deltaTime;
    
    // Apply damping
    agent.velocity.x *= this.config.damping;
    agent.velocity.y *= this.config.damping;
    agent.velocity.z *= this.config.damping;
    
    // Clamp velocity
    const velocityMagnitude = Math.sqrt(
      agent.velocity.x ** 2 + 
      agent.velocity.y ** 2 + 
      agent.velocity.z ** 2
    );
    
    if (velocityMagnitude > this.config.maxVelocity) {
      const scale = this.config.maxVelocity / velocityMagnitude;
      agent.velocity.x *= scale;
      agent.velocity.y *= scale;
      agent.velocity.z *= scale;
    }
    
    // Update position
    agent.position.x += agent.velocity.x * deltaTime;
    agent.position.y += agent.velocity.y * deltaTime;
    agent.position.z += agent.velocity.z * deltaTime;
    
    // Store force for visualization
    agent.force = force;
  }
  
  // Calculate interaction force between two agents
  calculateInteractionForce(agent, otherAgent) {
    const dx = otherAgent.position.x - agent.position.x;
    const dy = otherAgent.position.y - agent.position.y;
    const dz = otherAgent.position.z - agent.position.z;
    
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (distance === 0) return { x: 0, y: 0, z: 0 };
    
    // Normalize direction
    const dirX = dx / distance;
    const dirY = dy / distance;
    const dirZ = dz / distance;
    
    let forceMagnitude = 0;
    
    // Repulsion force (prevent overlap)
    if (distance < this.config.linkDistance) {
      forceMagnitude = -this.config.nodeRepulsion / (distance * distance + 1);
    }
    
    // Attraction force for connected agents
    const connection = agent.connections?.find(c => c.agentId === otherAgent.id);
    if (connection) {
      const attractionDistance = this.config.linkDistance * (1 + connection.strength);
      if (distance > attractionDistance) {
        forceMagnitude += this.config.springStrength * connection.strength * 
          (distance - attractionDistance);
      }
    }
    
    return {
      x: dirX * forceMagnitude,
      y: dirY * forceMagnitude,
      z: dirZ * forceMagnitude
    };
  }
  
  // Calculate hierarchical force (attraction to coordinators)
  calculateHierarchicalForce(agent, allAgents) {
    const force = { x: 0, y: 0, z: 0 };
    
    const leaders = allAgents.filter(a => 
      a.type === 'queen' || a.type === 'coordinator'
    );
    
    leaders.forEach(leader => {
      const dx = leader.position.x - agent.position.x;
      const dy = leader.position.y - agent.position.y;
      const dz = leader.position.z - agent.position.z;
      
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (distance === 0) return;
      
      const gravityForce = leader.type === 'queen' 
        ? this.config.queenGravity 
        : this.config.hierarchicalForce;
      
      const magnitude = gravityForce / (distance + 1);
      
      force.x += (dx / distance) * magnitude;
      force.y += (dy / distance) * magnitude;
      force.z += (dz / distance) * magnitude;
    });
    
    return force;
  }
  
  // Calculate gravity force towards center
  calculateGravityForce(position) {
    return {
      x: -position.x * this.config.gravityStrength,
      y: -position.y * this.config.gravityStrength,
      z: -position.z * this.config.gravityStrength
    };
  }
  
  // Calculate swarm cohesion force
  calculateSwarmCohesion(agent, allAgents) {
    // Find agents in the same swarm
    const swarmAgents = allAgents.filter(a => 
      a.metadata?.swarmId === agent.metadata?.swarmId && 
      a.id !== agent.id
    );
    
    if (swarmAgents.length === 0) return { x: 0, y: 0, z: 0 };
    
    // Calculate center of mass
    const centerOfMass = swarmAgents.reduce((acc, a) => ({
      x: acc.x + a.position.x,
      y: acc.y + a.position.y,
      z: acc.z + a.position.z
    }), { x: 0, y: 0, z: 0 });
    
    centerOfMass.x /= swarmAgents.length;
    centerOfMass.y /= swarmAgents.length;
    centerOfMass.z /= swarmAgents.length;
    
    // Calculate force towards center of mass
    const dx = centerOfMass.x - agent.position.x;
    const dy = centerOfMass.y - agent.position.y;
    const dz = centerOfMass.z - agent.position.z;
    
    return {
      x: dx * this.config.swarmCohesion,
      y: dy * this.config.swarmCohesion,
      z: dz * this.config.swarmCohesion
    };
  }
  
  // Apply message flow force
  applyMessageForce(fromAgent, toAgent, strength = 1.0) {
    const dx = toAgent.position.x - fromAgent.position.x;
    const dy = toAgent.position.y - fromAgent.position.y;
    const dz = toAgent.position.z - fromAgent.position.z;
    
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (distance === 0) return;
    
    const forceMagnitude = this.config.messageAttraction * strength;
    
    // Apply attractive force to both agents
    const fx = (dx / distance) * forceMagnitude;
    const fy = (dy / distance) * forceMagnitude;
    const fz = (dz / distance) * forceMagnitude;
    
    fromAgent.velocity.x += fx;
    fromAgent.velocity.y += fy;
    fromAgent.velocity.z += fz;
    
    toAgent.velocity.x -= fx;
    toAgent.velocity.y -= fy;
    toAgent.velocity.z -= fz;
  }
  
  // Get physics snapshot for visualization
  getSnapshot(agents) {
    return {
      frameCount: this.frameCount,
      timestamp: Date.now(),
      config: { ...this.config },
      agents: agents.map(agent => ({
        id: agent.id,
        position: { ...agent.position },
        velocity: { ...agent.velocity },
        force: { ...agent.force }
      }))
    };
  }
  
  // Calculate optimal physics parameters
  optimizeParameters(agents, targetMetric = 'balance') {
    const currentConfig = { ...this.config };
    const suggestions = {};
    
    // Analyze current state
    const avgVelocity = agents.reduce((sum, a) => 
      sum + Math.sqrt(a.velocity.x ** 2 + a.velocity.y ** 2 + a.velocity.z ** 2), 0
    ) / agents.length;
    
    const avgDistance = this.calculateAverageDistance(agents);
    
    switch (targetMetric) {
      case 'latency':
        // Optimize for fast message propagation
        if (avgDistance > this.config.linkDistance * 2) {
          suggestions.springStrength = this.config.springStrength * 1.2;
          suggestions.messageAttraction = this.config.messageAttraction * 1.5;
        }
        break;
        
      case 'throughput':
        // Optimize for maximum parallel communication
        if (avgDistance < this.config.linkDistance * 0.8) {
          suggestions.nodeRepulsion = this.config.nodeRepulsion * 1.3;
          suggestions.linkDistance = this.config.linkDistance * 1.1;
        }
        break;
        
      case 'balance':
        // Optimize for overall stability
        if (avgVelocity > this.config.maxVelocity * 0.5) {
          suggestions.damping = Math.min(0.99, this.config.damping * 1.05);
        }
        if (avgDistance > this.config.linkDistance * 1.5) {
          suggestions.gravityStrength = this.config.gravityStrength * 1.1;
        }
        break;
    }
    
    return {
      current: currentConfig,
      suggested: suggestions,
      metrics: {
        avgVelocity,
        avgDistance,
        stability: 1 - (avgVelocity / this.config.maxVelocity)
      }
    };
  }
  
  // Calculate average distance between connected agents
  calculateAverageDistance(agents) {
    let totalDistance = 0;
    let connectionCount = 0;
    
    agents.forEach(agent => {
      agent.connections?.forEach(connection => {
        const otherAgent = agents.find(a => a.id === connection.agentId);
        if (otherAgent) {
          const dx = otherAgent.position.x - agent.position.x;
          const dy = otherAgent.position.y - agent.position.y;
          const dz = otherAgent.position.z - agent.position.z;
          totalDistance += Math.sqrt(dx * dx + dy * dy + dz * dz);
          connectionCount++;
        }
      });
    });
    
    return connectionCount > 0 ? totalDistance / connectionCount : this.config.linkDistance;
  }
}