use std::collections::HashSet;

use crate::backprop::cosine;
use crate::encoder::Encoder;

pub const MIN_QUERY_CONFIDENCE: f32 = 0.25;
const MAX_ANSWER_WORDS: usize = 6;
const STRONG_PHRASE_CONFIDENCE: f32 = 0.92;

#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAnswer {
    pub answer: String,
    pub confidence: f32,
}

pub fn decode_answer(output: &[f32], encoder: &Encoder, question: &str) -> Option<DecodedAnswer> {
    if output.iter().all(|value| value.abs() <= f32::EPSILON) {
        return None;
    }

    let query_words = normalized_words(question)
        .into_iter()
        .collect::<HashSet<_>>();
    let mut candidates = encoder
        .known_words()
        .into_iter()
        .filter(|word| !query_words.contains(word))
        .filter(|word| !is_stopword(word))
        .filter_map(|word| {
            let vector = encoder.encode_deterministic(&word);
            if vector.iter().all(|value| value.abs() <= f32::EPSILON) {
                return None;
            }
            let score = cosine(output, &vector);
            score.is_finite().then_some((word, score))
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (best_word, best_score) = candidates
        .first()
        .map(|(word, score)| (word.clone(), *score))?;
    if best_score < MIN_QUERY_CONFIDENCE {
        return None;
    }

    if best_score >= STRONG_PHRASE_CONFIDENCE && should_return_best_candidate_only(&best_word) {
        return Some(DecodedAnswer {
            answer: best_word,
            confidence: best_score.clamp(0.0, 1.0),
        });
    }

    let threshold = (best_score * 0.55).max(MIN_QUERY_CONFIDENCE * 0.75);
    let mut words = candidates
        .iter()
        .filter(|(_, score)| *score >= threshold)
        .take(MAX_ANSWER_WORDS)
        .map(|(word, _)| word.clone())
        .collect::<Vec<_>>();

    if words.len() < 3 {
        for (word, score) in &candidates {
            if words.len() >= 3 || *score <= 0.0 {
                break;
            }
            if !words.contains(word) {
                words.push(word.clone());
            }
        }
    }

    if words.is_empty() {
        return None;
    }

    Some(DecodedAnswer {
        answer: words.join(" "),
        confidence: best_score.clamp(0.0, 1.0),
    })
}

fn should_return_best_candidate_only(candidate: &str) -> bool {
    contains_cjk(candidate) || candidate.chars().count() > 12
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

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(|raw| {
            let cleaned = raw
                .chars()
                .filter(|ch| ch.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();

            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned)
            }
        })
        .collect()
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "the"
            | "to"
            | "was"
            | "were"
            | "what"
            | "when"
            | "where"
            | "who"
            | "why"
            | "with"
    )
}
