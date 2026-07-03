use std::collections::HashMap;

const DEFAULT_MAX_NGRAM: usize = 4;

/// Character prefix n-gram tokenizer for Stage 5.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub next_id: u32,
    pub max_ngram: usize,
}

impl Tokenizer {
    pub fn new(max_ngram: usize) -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            next_id: 0,
            max_ngram: max_ngram.max(1),
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        for term in normalized_terms(text) {
            for token in tokens_for_term(&term, self.max_ngram) {
                ids.push(self.get_or_insert(token));
            }
        }

        ids
    }

    pub fn encode_deterministic(&self, text: &str) -> Vec<u32> {
        normalized_terms(text)
            .into_iter()
            .flat_map(|term| tokens_for_term(&term, self.max_ngram))
            .filter_map(|token| self.vocab.get(&token).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let boundary_tokens = ids
            .iter()
            .filter_map(|id| self.id_to_token.get(id))
            .filter_map(|token| token.strip_prefix('#'))
            .collect::<Vec<_>>();

        if !boundary_tokens.is_empty() {
            return boundary_tokens.join(" ");
        }

        ids.iter()
            .filter_map(|id| self.id_to_token.get(id))
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab_size(&self) -> u32 {
        self.next_id
    }

    fn get_or_insert(&mut self, token: String) -> u32 {
        if let Some(id) = self.vocab.get(&token) {
            return *id;
        }

        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.vocab.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
        id
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_NGRAM)
    }
}

fn normalized_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lowered in ch.to_lowercase() {
                current.push(lowered);
            }
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        terms.push(current);
    }

    terms
}

fn tokens_for_term(term: &str, max_ngram: usize) -> Vec<String> {
    if contains_cjk(term) {
        cjk_ngrams_for_term(term, max_ngram)
    } else {
        ngrams_for_word(term, max_ngram)
    }
}

fn cjk_ngrams_for_term(term: &str, max_ngram: usize) -> Vec<String> {
    let chars = term.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();

    for start in 0..chars.len() {
        let max_len = max_ngram.min(chars.len() - start);
        for len in 1..=max_len {
            tokens.push(chars[start..start + len].iter().collect());
        }
    }

    tokens.push(format!("#{term}"));
    tokens
}

fn contains_cjk(text: &str) -> bool {
    text.chars().any(is_cjk)
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch,
        '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{F900}'..='\u{FAFF}'
            | '\u{20000}'..='\u{2A6DF}'
            | '\u{2A700}'..='\u{2B73F}'
            | '\u{2B740}'..='\u{2B81F}'
            | '\u{2B820}'..='\u{2CEAF}'
            | '\u{2CEB0}'..='\u{2EBEF}'
    )
}

fn ngrams_for_word(word: &str, max_ngram: usize) -> Vec<String> {
    let chars = word.chars().collect::<Vec<_>>();
    let prefix_count = max_ngram.min(chars.len());
    let mut tokens = Vec::with_capacity(prefix_count + 1);

    for len in 1..=prefix_count {
        tokens.push(chars[..len].iter().collect());
    }
    tokens.push(format!("#{word}"));

    tokens
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn cat_and_cats_share_tokens() {
        let mut tokenizer = Tokenizer::new(4);
        let cat_ids = tokenizer.encode("cat");
        let cats_ids = tokenizer.encode("cats");

        let cat_set = cat_ids.into_iter().collect::<HashSet<_>>();
        let cats_set = cats_ids.into_iter().collect::<HashSet<_>>();
        let shared = cat_set.intersection(&cats_set).count();

        assert!(
            shared >= 3,
            "cat and cats should share at least 3 tokens, got {shared}"
        );
    }

    #[test]
    fn encode_is_deterministic() {
        let mut tokenizer = Tokenizer::new(4);

        let first = tokenizer.encode("the quick brown fox");
        let second = tokenizer.encode("the quick brown fox");

        assert_eq!(first, second);
    }

    #[test]
    fn empty_string_returns_empty() {
        let mut tokenizer = Tokenizer::new(4);

        assert!(tokenizer.encode("").is_empty());
        assert!(tokenizer.encode("!? --").is_empty());
    }

    #[test]
    fn vocab_grows_on_new_words() {
        let mut tokenizer = Tokenizer::new(4);
        let before = tokenizer.vocab_size();

        tokenizer.encode("xyzqwerty");

        assert!(tokenizer.vocab_size() > before);
    }

    #[test]
    fn decode_encode_roundtrip_reasonable() {
        let mut tokenizer = Tokenizer::new(4);

        let ids = tokenizer.encode("rust programming");
        let decoded = tokenizer.decode(&ids);

        assert!(
            decoded.contains("rust") || decoded.contains("programming"),
            "decoded '{decoded}' does not resemble original"
        );
    }

    #[test]
    fn encode_deterministic_does_not_grow_vocab() {
        let mut tokenizer = Tokenizer::new(4);
        tokenizer.encode("cat");
        let before = tokenizer.vocab_size();

        let ids = tokenizer.encode_deterministic("cat dog");

        assert!(!ids.is_empty());
        assert_eq!(tokenizer.vocab_size(), before);
        assert!(tokenizer.encode_deterministic("dog").is_empty());
    }

    #[test]
    fn punctuation_and_case_normalize_to_same_ids() {
        let mut tokenizer = Tokenizer::new(4);

        let first = tokenizer.encode("Cat!");
        let second = tokenizer.encode("cat");

        assert_eq!(first, second);
    }

    #[test]
    fn unicode_alphanumeric_text_is_tokenized_by_char() {
        let mut tokenizer = Tokenizer::new(4);

        let ids = tokenizer.encode("मनस्");
        let decoded = tokenizer.decode(&ids);

        assert!(!ids.is_empty());
        assert_eq!(decoded, "मनस");
    }

    #[test]
    fn chinese_text_uses_sliding_character_ngrams() {
        let mut tokenizer = Tokenizer::new(4);

        let ids = tokenizer.encode("山是由构造力形成的");
        let tokens = ids
            .iter()
            .filter_map(|id| tokenizer.id_to_token.get(id))
            .cloned()
            .collect::<HashSet<_>>();

        assert!(tokens.contains("山"));
        assert!(tokens.contains("构造"));
        assert!(tokens.contains("形成"));
        assert!(tokens.contains("#山是由构造力形成的"));
    }

    #[test]
    fn chinese_related_phrases_share_tokens() {
        let mut tokenizer = Tokenizer::new(4);

        let first = tokenizer.encode("山是怎么形成的");
        let second = tokenizer.encode("山是由构造力形成的");

        let first_set = first.into_iter().collect::<HashSet<_>>();
        let second_set = second.into_iter().collect::<HashSet<_>>();
        assert!(first_set.intersection(&second_set).count() >= 3);
    }
}
