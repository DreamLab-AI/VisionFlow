//! Prompt templates for triple extraction in multiple languages

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Triple extraction instructions for different languages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleInstructions {
    instructions: HashMap<String, LanguageInstructions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInstructions {
    pub system: String,
    pub entity_relation: String,
    pub event_entity: String,
    pub event_relation: String,
    pub passage_start: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptInstructions {
    instructions: HashMap<String, ConceptLanguageInstructions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptLanguageInstructions {
    pub event: String,
    pub entity: String,
    pub relation: String,
}

impl Default for TripleInstructions {
    fn default() -> Self {
        Self::new()
    }
}

impl TripleInstructions {
    pub fn new() -> Self {
        let mut instructions = HashMap::new();

        // English instructions
        instructions.insert("en".to_string(), LanguageInstructions {
            system: "You are a helpful assistant who always response in a valid array of JSON objects without any explanation".to_string(),
            entity_relation: r#"Given a passage, summarize all the important entities and the relations between them in a concise manner. Relations should briefly capture the connections between entities, without repeating information from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from being considered as entities.
        You must **strictly output in the following JSON format**:

        [
            {
                "Head": "{a noun}",
                "Relation": "{a verb}",
                "Tail": "{a noun}",
            }...
        ]"#.to_string(),
            event_entity: r#"Please analyze and summarize the participation relations between the events and entities in the given paragraph. Each event is a single independent sentence. Additionally, identify all the entities that participated in the events. Do not use ellipses.
        You must **strictly output in the following JSON format**:

        [
            {
                "Event": "{a simple sentence describing an event}",
                "Entity": ["entity 1", "entity 2", "..."]
            }...
        ]"#.to_string(),
            event_relation: r#"Please analyze and summarize the relationships between the events in the paragraph. Each event is a single independent sentence. Identify temporal and causal relationships between the events using the following types: before, after, at the same time, because, and as a result. Each extracted triple should be specific, meaningful, and able to stand alone.  Do not use ellipses.
        You must **strictly output in the following JSON format**:

        [
            {
                "Head": "{a simple sentence describing the event 1}",
                "Relation": "{temporal or causality relation between the events}",
                "Tail": "{a simple sentence describing the event 2}"
            }...
        ]"#.to_string(),
            passage_start: "Here is the passage.".to_string(),
        });

        // Chinese Simplified instructions
        instructions.insert("zh-CN".to_string(), LanguageInstructions {
            system: "你是一个始终以有效JSON数组格式回应的助手".to_string(),
            entity_relation: r#"给定一段文字，提取所有重要实体及其关系，并以简洁的方式总结。关系描述应清晰表达实体间的联系，且不重复头尾实体的信息。实体需具体明确，排除代词。
        返回格式必须为以下JSON结构,内容需用简体中文表述:
        [
            {
                "Head": "{名词}",
                "Relation": "{动词或关系描述}",
                "Tail": "{名词}"
            }...
        ]"#.to_string(),
            event_entity: r#"分析段落中的事件及其参与实体。每个事件应为独立单句，列出所有相关实体（需具体，不含代词）。
        返回格式必须为以下JSON结构,内容需用简体中文表述:
        [
            {
                "Event": "{描述事件的简单句子}",
                "Entity": ["实体1", "实体2", "..."]
            }...
        ]"#.to_string(),
            event_relation: r#"分析事件间的时序或因果关系,关系类型包括:之前,之后,同时,因为,结果.每个事件应为独立单句。
        返回格式必须为以下JSON结构.内容需用简体中文表述.
        [
            {
                "Head": "{事件1描述}",
                "Relation": "{时序/因果关系}",
                "Tail": "{事件2描述}"
            }...
        ]"#.to_string(),
            passage_start: "给定以下段落：".to_string(),
        });

        // Chinese Traditional instructions
        instructions.insert("zh-HK".to_string(), LanguageInstructions {
            system: "你是一個始終以有效JSON數組格式回覆的助手".to_string(),
            entity_relation: r#"給定一段文字，提取所有重要實體及其關係，並以簡潔的方式總結。關係描述應清晰表達實體間的聯繫，且不重複頭尾實體的信息。實體需具體明確，排除代詞。
        返回格式必須為以下JSON結構,內容需用繁體中文表述:
        [
            {
                "Head": "{名詞}",
                "Relation": "{動詞或關係描述}",
                "Tail": "{名詞}"
            }...
        ]"#.to_string(),
            event_entity: r#"分析段落中的事件及其參與實體。每個事件應為獨立單句，列出所有相關實體（需具體，不含代詞）。
        返回格式必須為以下JSON結構,內容需用繁體中文表述:
        [
            {
                "Event": "{描述事件的簡單句子}",
                "Entity": ["實體1", "實體2", "..."]
            }...
        ]"#.to_string(),
            event_relation: r#"分析事件間的時序或因果關係,關係類型包括:之前,之後,同時,因為,結果.每個事件應為獨立單句。
        返回格式必須為以下JSON結構.內容需用繁體中文表述.
        [
            {
                "Head": "{事件1描述}",
                "Relation": "{時序/因果關係}",
                "Tail": "{事件2描述}"
            }...
        ]"#.to_string(),
            passage_start: "給定以下段落：".to_string(),
        });

        Self { instructions }
    }

    /// Get instructions for a specific language
    pub fn get_instructions(&self, language: &str) -> Option<&LanguageInstructions> {
        self.instructions.get(language)
    }

    /// Get instructions for a language, falling back to English
    pub fn get_instructions_or_default(&self, language: &str) -> &LanguageInstructions {
        self.instructions.get(language)
            .or_else(|| self.instructions.get("en"))
            .expect("English instructions must be available")
    }

    /// Get supported languages
    pub fn supported_languages(&self) -> Vec<&str> {
        self.instructions.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a language is supported
    pub fn is_language_supported(&self, language: &str) -> bool {
        self.instructions.contains_key(language)
    }

    /// Add custom language instructions
    pub fn add_language(&mut self, language: String, instructions: LanguageInstructions) {
        self.instructions.insert(language, instructions);
    }
}

impl Default for ConceptInstructions {
    fn default() -> Self {
        Self::new()
    }
}

impl ConceptInstructions {
    pub fn new() -> Self {
        let mut instructions = HashMap::new();

        // English concept instructions
        instructions.insert("en".to_string(), ConceptLanguageInstructions {
            event: r#"I will give you an EVENT. You need to give several phrases containing 1-2 words for the ABSTRACT EVENT of this EVENT.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract event words should fulfill the following requirements.
            1. The ABSTRACT EVENT phrases can well represent the EVENT, and it could be the type of the EVENT or the related concepts of the EVENT.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            EVENT: A man retreats to mountains and forests.
            Your answer: retreat, relaxation, escape, nature, solitude
            EVENT: A cat chased a prey into its shelter
            Your answer: hunting, escape, predation, hidding, stalking
            EVENT: Sam playing with his dog
            Your answer: relaxing event, petting, playing, bonding, friendship
            EVENT: [EVENT]
            Your answer:"#.to_string(),
            entity: r#"I will give you an ENTITY. You need to give several phrases containing 1-2 words for the ABSTRACT ENTITY of this ENTITY.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT ENTITY phrases can well represent the ENTITY, and it could be the type of the ENTITY or the related concepts of the ENTITY.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            ENTITY: Soul
            CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release
            Your answer: movie, film

            ENTITY: Thinkpad X60
            CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60
            Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand

            ENTITY: [ENTITY]
            CONTEXT: [CONTEXT]
            Your answer:"#.to_string(),
            relation: r#"I will give you an RELATION. You need to give several phrases containing 1-2 words for the ABSTRACT RELATION of this RELATION.
            You must return your answer in the following format: phrases1, phrases2, phrases3,...
            You can't return anything other than answers.
            These abstract intention words should fulfill the following requirements.
            1. The ABSTRACT RELATION phrases can well represent the RELATION, and it could be the type of the RELATION or the simplest concepts of the RELATION.
            2. Strictly follow the provided format, do not add extra characters or words.
            3. Write at least 3 or more phrases at different abstract level if possible.
            4. Do not repeat the same word and the input in the answer.
            5. Stop immediately if you can't think of any more phrases, and no explanation is needed.

            RELATION: participated in
            Your answer: become part of, attend, take part in, engage in, involve in
            RELATION: be included in
            Your answer: join, be a part of, be a member of, be a component of
            RELATION: [RELATION]
            Your answer:"#.to_string(),
        });

        Self { instructions }
    }

    /// Get concept instructions for a specific language
    pub fn get_instructions(&self, language: &str) -> Option<&ConceptLanguageInstructions> {
        self.instructions.get(language)
    }

    /// Get concept instructions for a language, falling back to English
    pub fn get_instructions_or_default(&self, language: &str) -> &ConceptLanguageInstructions {
        self.instructions.get(language)
            .or_else(|| self.instructions.get("en"))
            .expect("English concept instructions must be available")
    }
}

/// Processing stage for triple extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingStage {
    EntityRelation,
    EventEntity,
    EventRelation,
}

impl ProcessingStage {
    /// Get the prompt for this stage
    pub fn get_prompt(self, instructions: &LanguageInstructions) -> &str {
        match self {
            ProcessingStage::EntityRelation => &instructions.entity_relation,
            ProcessingStage::EventEntity => &instructions.event_entity,
            ProcessingStage::EventRelation => &instructions.event_relation,
        }
    }

    /// Get all processing stages in order
    pub fn all() -> [ProcessingStage; 3] {
        [
            ProcessingStage::EntityRelation,
            ProcessingStage::EventEntity,
            ProcessingStage::EventRelation,
        ]
    }

    /// Get stage name as string
    pub fn as_str(self) -> &'static str {
        match self {
            ProcessingStage::EntityRelation => "entity_relation",
            ProcessingStage::EventEntity => "event_entity",
            ProcessingStage::EventRelation => "event_relation",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triple_instructions_creation() {
        let instructions = TripleInstructions::new();
        assert!(instructions.is_language_supported("en"));
        assert!(instructions.is_language_supported("zh-CN"));
        assert!(instructions.is_language_supported("zh-HK"));
        assert!(!instructions.is_language_supported("fr"));
    }

    #[test]
    fn test_get_instructions() {
        let instructions = TripleInstructions::new();
        let en_instructions = instructions.get_instructions("en").unwrap();
        assert!(!en_instructions.system.is_empty());
        assert!(!en_instructions.entity_relation.is_empty());
    }

    #[test]
    fn test_get_instructions_or_default() {
        let instructions = TripleInstructions::new();
        let default_instructions = instructions.get_instructions_or_default("unknown");
        let en_instructions = instructions.get_instructions("en").unwrap();
        assert_eq!(default_instructions.system, en_instructions.system);
    }

    #[test]
    fn test_processing_stages() {
        let stages = ProcessingStage::all();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0], ProcessingStage::EntityRelation);
        assert_eq!(stages[1], ProcessingStage::EventEntity);
        assert_eq!(stages[2], ProcessingStage::EventRelation);
    }

    #[test]
    fn test_stage_as_str() {
        assert_eq!(ProcessingStage::EntityRelation.as_str(), "entity_relation");
        assert_eq!(ProcessingStage::EventEntity.as_str(), "event_entity");
        assert_eq!(ProcessingStage::EventRelation.as_str(), "event_relation");
    }
}