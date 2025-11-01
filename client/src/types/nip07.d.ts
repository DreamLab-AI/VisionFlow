// Type definitions for NIP-07 window.nostr based on the specification
// https://github.com/nostr-protocol/nips/blob/master/07.md

import type { Event as NostrEvent, UnsignedEvent } from 'nostr-tools';

// Define the structure of the event object passed to signEvent
// Note: NIP-07 specifies the input event lacks id, pubkey, sig.
// nostr-tools' UnsignedEvent fits this description.
type Nip07Event = Omit<UnsignedEvent, 'pubkey'>; 

// Define the interface for the window.nostr object
interface NostrProvider {
  getPublicKey(): Promise<string>; 
  signEvent(event: Nip07Event): Promise<NostrEvent>; 

  
  nip44?: {
    encrypt(pubkey: string, plaintext: string): Promise<string>; 
    decrypt(pubkey: string, ciphertext: string): Promise<string>; 
  };

  
  nip04?: {
    encrypt(pubkey: string, plaintext: string): Promise<string>;
    decrypt(pubkey: string, ciphertext: string): Promise<string>;
  };

  
  getRelays?(): Promise<{ [url: string]: { read: boolean; write: boolean } }>;
}

// Augment the global Window interface
declare global {
  interface Window {
    nostr?: NostrProvider;
  }
}

// Export an empty object to ensure this is treated as a module
export {};