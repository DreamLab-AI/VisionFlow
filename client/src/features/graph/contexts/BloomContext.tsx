import React, { createContext, useContext } from 'react';

interface BloomContextValue {
  envBloomStrength: number;
  nodeBloomStrength: number;
  edgeBloomStrength: number;
}

const BloomContext = createContext<BloomContextValue>({
  envBloomStrength: 1,
  nodeBloomStrength: 1,
  edgeBloomStrength: 1,
});

export const useBloomStrength = () => useContext(BloomContext);

export const BloomProvider: React.FC<{
  children: React.ReactNode;
  envBloomStrength: number;
  nodeBloomStrength: number;
  edgeBloomStrength: number;
}> = ({ children, envBloomStrength, nodeBloomStrength, edgeBloomStrength }) => {
  return (
    <BloomContext.Provider value={{ envBloomStrength, nodeBloomStrength, edgeBloomStrength }}>
      {children}
    </BloomContext.Provider>
  );
};