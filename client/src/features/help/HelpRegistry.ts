import { HelpContent, HelpCategory } from './types';

export class HelpRegistry {
  private helpItems: Map<string, HelpContent> = new Map();
  private categories: Map<string, HelpCategory> = new Map();
  private listeners: Set<() => void> = new Set();

  
  registerHelp(help: HelpContent): void {
    this.helpItems.set(help.id, help);
    this.notifyListeners();
  }

  registerBulkHelp(helpItems: HelpContent[]): void {
    helpItems.forEach(item => this.helpItems.set(item.id, item));
    this.notifyListeners();
  }

  
  registerCategory(category: HelpCategory): void {
    this.categories.set(category.id, category);
    this.notifyListeners();
  }

  
  getHelp(id: string): HelpContent | undefined {
    return this.helpItems.get(id);
  }

  getAllHelp(): HelpContent[] {
    return Array.from(this.helpItems.values());
  }

  getCategory(id: string): HelpCategory | undefined {
    return this.categories.get(id);
  }

  getAllCategories(): HelpCategory[] {
    return Array.from(this.categories.values());
  }

  
  searchHelp(query: string): HelpContent[] {
    const lowerQuery = query.toLowerCase();
    return this.getAllHelp().filter(item => 
      item.title.toLowerCase().includes(lowerQuery) ||
      item.description.toLowerCase().includes(lowerQuery) ||
      item.detailedHelp?.toLowerCase().includes(lowerQuery) ||
      item.relatedTopics?.some(topic => topic.toLowerCase().includes(lowerQuery))
    );
  }

  
  addListener(listener: () => void): void {
    this.listeners.add(listener);
  }

  removeListener(listener: () => void): void {
    this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => listener());
  }
}

// Global instance
export const helpRegistry = new HelpRegistry();