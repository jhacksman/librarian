import test from 'node:test';
import assert from 'node:assert/strict';
import { validateAnswerSupport, numberSourceLines } from '../src/answer-support.mjs';

const evidence = [{ bookId: 'source-a', sectionId: 'page-1', text: 'COUNT(*) counts every row including NULLs.',
  locator: { page: 1, charStart: 10, charEnd: 50, offsetBasis: 'section_utf16_code_units' } },
{ bookId: 'source-b', sectionId: 'page-9', text: 'COUNT(column) counts only non-NULL entries.',
  locator: { page: 9, charStart: 0, charEnd: 41, offsetBasis: 'section_utf16_code_units' } }];

test('support references resolve exact quotation offsets and preserve distinct source identities', () => {
  const result = validateAnswerSupport({ abstain: false, answer: 'Count all rows [1].\n\nCount populated entries [2].',
    support: [{ paragraph: 1, citation: 1, quote: 'counts every row including NULLs.' },
      { paragraph: 2, citation: 2, quote: 'counts only non-NULL entries.' }] }, evidence);
  assert.equal(result.error, undefined);
  assert.equal(result.support[0].locator.charStart, 19);
  assert.equal(result.support[1].bookId, 'source-b');
  assert.equal(result.semanticSupport, 'not_automatically_verified');
});

test('absent quotes, wrong-source support and unsupported citations cannot pass source validation', () => {
  for (const support of [undefined, [], [{ paragraph: 1, citation: 1, quote: evidence[1].text }],
    [{ paragraph: 2, citation: 1, quote: evidence[0].text }],
    [{ paragraph: 1, citation: 2, quote: evidence[1].text }]]) {
    const result = validateAnswerSupport({ abstain: false, answer: 'Every row is counted. [1]', support }, evidence);
    assert.equal(result.error, 'invalid_support');
  }
  assert.equal(validateAnswerSupport({ abstain: false, answer: 'Every row is counted. [99]', support: [] }, evidence).error, 'invalid_citations');
  assert.deepEqual(validateAnswerSupport({ abstain: true, answer: '', support: [] }, evidence), { abstain: true });
});

test('exact quotation membership explicitly does not certify semantic entailment', () => {
  const result = validateAnswerSupport({ abstain: false, answer: 'The moon is made of cheese. [1]',
    support: [{ paragraph: 1, citation: 1, quote: evidence[0].text }] }, evidence);
  assert.equal(result.error, undefined);
  assert.equal(result.semanticSupport, 'not_automatically_verified');
});

test('structured paragraphs render citations only after exact source support validates', () => {
  const result = validateAnswerSupport({ abstain: false, paragraphs: [
    { text: 'Count every row.', support: [{ citation: 1, quote: 'counts every row including NULLs.' }] },
    { text: 'Count populated entries instead.', support: [
      { citation: 2, quote: 'counts only non-NULL entries.' },
      { citation: 2, quote: 'COUNT(column) counts only non-NULL entries.' }] },
  ] }, evidence);
  assert.equal(result.error, undefined);
  assert.equal(result.answer, 'Count every row. [1]\n\nCount populated entries instead. [2]');
  assert.deepEqual(result.references, [1, 2]);
  assert.deepEqual(result.support.map(item => item.paragraph), [1, 2, 2]);
  assert.equal(result.support[0].locator.charStart, 19);
  assert.equal(result.support[1].bookId, 'source-b');
  assert.equal(result.semanticSupport, 'not_automatically_verified');
  // The old invalid response remains invalid. This is a new output protocol,
  // not a post-hoc acceptance of the retained markerless model responses.
  assert.equal(validateAnswerSupport({ abstain: false, answer: 'Count every row.',
    support: [{ paragraph: 1, citation: 1, quote: evidence[0].text }] }, evidence).error, 'invalid_citations');
});

test('structured paragraph support cannot bypass missing, fabricated or wrong-source quotations', () => {
  for (const support of [[], null, [{ citation: 99, quote: evidence[0].text }],
    [{ citation: 1, quote: evidence[1].text }], [{ citation: 1, quote: 'A fabricated source quotation.' }],
    [{ citation: 1, quote: 'short' }], [{ citation: 1.5, quote: evidence[0].text }]]) {
    assert.equal(validateAnswerSupport({ abstain: false, paragraphs: [
      { text: 'Count every row.', support }] }, evidence).error, 'invalid_support');
  }
  assert.equal(validateAnswerSupport({ abstain: false, paragraphs: [
    { text: 'Count every row.', support: [{ citation: 1, quote: evidence[0].text }] },
    { text: 'Also do unrelated work.', support: [] }] }, evidence).error, 'invalid_support');
});

test('structured abstention and paragraph boundaries are explicit and bounded', () => {
  assert.deepEqual(validateAnswerSupport({ abstain: true, paragraphs: [] }, evidence), { abstain: true });
  const paragraph = { text: 'Count every row.', support: [{ citation: 1, quote: evidence[0].text }] };
  for (const value of [{ abstain: true, paragraphs: [paragraph] }, { abstain: false, paragraphs: [] },
    { abstain: false, paragraphs: [paragraph], answer: 'Conflicting answer.' },
    { abstain: false, paragraphs: [paragraph], support: [] },
    { abstain: false, paragraphs: Array(25).fill(paragraph) },
    ...['', 'Count rows. [99]', 'Count rows.\n\nUnrelated claim.'].map(text =>
      ({ abstain: false, paragraphs: [{ ...paragraph, text }] }))]) {
    assert.equal(validateAnswerSupport(value, evidence).error, 'invalid_answer');
  }
  assert.equal(validateAnswerSupport({ abstain: false, paragraphs: [
    { ...paragraph, support: Array(97).fill(paragraph.support[0]) }] }, evidence).error, 'invalid_support');
});


test('numbered line support retains exact whitespace, newlines, and UTF-16 occurrence offsets',()=> {
  const text='Lead 😀\nRepeated exact supporting line.\n\nRepeated exact supporting line.\nCASE WHEN count(x) = 0 THEN 1\n  ELSE count(x) END';
  const source=[{bookId:'book',sectionId:'section',text,locator:{charStart:80,charEnd:80+text.length}}];
  const prompt=numberSourceLines(text);assert.equal(prompt.split('\n')[2],'3|');
  const result=validateAnswerSupport({abstain:false,paragraphs:[{text:'The guard substitutes one for zero.',support:[{citation:1,lines:[5,6]}]},
    {text:'Repeated source wording.',support:[{citation:1,lines:[4,4]}]}]},source);
  assert.equal(result.error,undefined);
  assert.equal(result.support[0].quote,'CASE WHEN count(x) = 0 THEN 1\n  ELSE count(x) END');
  assert.equal(result.support[0].locator.charStart,80+text.indexOf('CASE'));
  assert.equal(result.support[1].locator.charStart,80+text.lastIndexOf('Repeated'));
  for(const item of result.support)assert.equal(text.slice(item.locator.charStart-80,item.locator.charEnd-80),item.quote);
  assert.equal(result.semanticSupport,'not_automatically_verified');
});

test('line support refuses out-of-range, mixed, empty, short, or oversized source selections',()=> {
  const source=[{bookId:'book',sectionId:'section',text:'A source statement long enough.\n\nshort\n'+'x'.repeat(2001),locator:{charStart:0}}];
  for(const reference of [{citation:1,lines:[0,1]},{citation:1,lines:[2,1]},{citation:1,lines:[1,5]},
    {citation:1,lines:[1.5,2]},{citation:1,lines:[1]},{citation:1,lines:[2,2]},
    {citation:1,lines:[3,3]},{citation:1,lines:[4,4]},{citation:2,lines:[1,1]},
    {citation:1,lines:[1,1],quote:source[0].text}]) {
    assert.equal(validateAnswerSupport({abstain:false,paragraphs:[{text:'A supported claim.',support:[reference]}]},source).error,'invalid_support');
  }
  // The original malformed recopy remains invalid, even if its meaning is right.
  assert.equal(validateAnswerSupport({abstain:false,paragraphs:[{text:'A supported claim.',support:[{citation:1,quote:'A source statement ... long enough.'}]}]},source).error,'invalid_support');
});
