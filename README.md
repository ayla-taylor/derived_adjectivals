# derived_adjective_events

This code is all related to my Capstone project for graduating the Brandeis Univeristy Computational Linguistics MS program. 

I am working with Dr. James Pustejovsky to determine what effect, if any, encoding event information into deverbal adjectives has on question answering. 

So far: 
- I have three groups of adjectives I am looking at: 
	1. 0-order derived adjectives: 'normal' root adjectives, that are not derived. Example: 'hard'
	2. 1st order derived adjectives: root verb adjectives; adjectives that are derived from verbs. Example: 'chopped' as in 'the chopped onions'
	3. 2nd order derived adjectives: derived root adjective; adjectives that are derived from verbs that were derived from adjectives. Example: 'hardened'
- The biggest challenge at this point is identifying sentences that have these constructions, as they are difficult to distinguish from verbs, particularly passive verbs. 
	- I am focusing specially on these adjectives in the prenomial position, as they are easier to idenify than in the predicative position. 
	- The data comes from the C4 corpus and includes all passages in which the target words occur in any capacity. It is processed with stanza, with the tokenize, POS tagging, lemmatizing, and dependency parsing processors. 
	- these processed sentences are then ran through the `get_derived.py` script, which pulls out the derived adjective positions