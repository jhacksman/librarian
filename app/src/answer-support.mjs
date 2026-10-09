// Exact quotes establish source identity, not semantic entailment of a claim.
export function numberSourceLines(text) {
  return text.split('\n').map((line,index)=>`${index+1}|${line}`).join('\n');
}
function lineQuote(reference, source) {
  if(!Array.isArray(reference.lines)||reference.lines.length!==2)return null;
  const [first,last]=reference.lines;
  const lines=source.text.split('\n');
  if(!Number.isSafeInteger(first)||!Number.isSafeInteger(last)||first<1||last<first||last>lines.length)return null;
  const offset=lines.slice(0,first-1).reduce((sum,line)=>sum+line.length+1,0);
  return {quote:lines.slice(first-1,last).join('\n'),offset};
}
function structuredParagraphs(parsed, evidence) {
  // A paragraph owns its support. Citation markers are presentation derived
  // from validated references, never a second model-generated copy of them.
  if (typeof parsed.abstain !== 'boolean' || Object.hasOwn(parsed, 'answer') || Object.hasOwn(parsed, 'support') ||
      !Array.isArray(parsed.paragraphs) || parsed.paragraphs.length > 24) {
    return { error: 'invalid_answer', message: 'Local AI did not return the required paragraph structure' };
  }
  if (parsed.abstain) return parsed.paragraphs.length === 0 ? { abstain: true } :
    { error: 'invalid_answer', message: 'Local AI returned paragraphs with an abstention' };
  if (!parsed.paragraphs.length) return { error: 'invalid_answer', message: 'Local AI returned an empty answer' };
  const paragraphs = [], support = [], offsets = [];
  for (const [index, item] of parsed.paragraphs.entries()) {
    if (!item || typeof item.text !== 'string' || !item.text.trim() || item.text.length > 12000 ||
        /\n\s*\n|\[\d+\]/.test(item.text)) {
      return { error: 'invalid_answer', message: 'Local AI returned an invalid paragraph' };
    }
    if (!Array.isArray(item.support) || !item.support.length || support.length + item.support.length > 96 ||
        item.support.some(reference => !reference || !Number.isSafeInteger(reference.citation) ||
          reference.citation < 1 || reference.citation > evidence.length)) {
      return { error: 'invalid_support', message: 'Every paragraph needs valid source support; its answer was withheld.' };
    }
    for(const reference of item.support) {
      const hasLines=Object.hasOwn(reference,'lines');
      if(hasLines&&Object.hasOwn(reference,'quote'))return {error:'invalid_support',message:'Use one support format per reference; its answer was withheld.'};
      const selected=hasLines?lineQuote(reference,evidence[reference.citation-1]):null;
      if((hasLines&&!selected)||(!hasLines&&typeof reference.quote!=='string'))return {error:'invalid_support',message:'Invalid source-line range; its answer was withheld.'};
      support.push({paragraph:index+1,citation:reference.citation,quote:hasLines?selected.quote:reference.quote});
      offsets.push(hasLines?selected.offset:null);
    }
    const references = [...new Set(item.support.map(reference => reference.citation))];
    paragraphs.push(`${item.text.trim()} ${references.map(citation => `[${citation}]`).join(' ')}`);
  }
  // Reuse the exact same quote membership, source identity and UTF-16 checks
  // as the legacy format below. No historic invalid answer is normalized.
  const checked=validateAnswerSupport({abstain:false,answer:paragraphs.join('\n\n'),support},evidence);
  if(!checked.error&&checked.support)for(const [index,item] of checked.support.entries()) {
    // A repeated line must point to the selected occurrence, not the first one.
    if(offsets[index]!==null) {
      const source=evidence[item.citation-1];
      item.locator={...source.locator,charStart:source.locator.charStart+offsets[index],charEnd:source.locator.charStart+offsets[index]+item.quote.length};
    }
  }
  return checked;
}

export function validateAnswerSupport(parsed, evidence) {
  if (parsed && Object.hasOwn(parsed, 'paragraphs')) return structuredParagraphs(parsed, evidence);
  if (!parsed || typeof parsed.abstain !== 'boolean' || typeof parsed.answer !== 'string') {
    return { error: 'invalid_answer', message: 'Local AI did not return the required answer structure' };
  }
  if (parsed.abstain) return { abstain: true };
  const answer = parsed.answer.trim();
  const paragraphs = answer.split(/\n\s*\n/).filter(part => part.trim());
  const references = [...answer.matchAll(/\[(\d+)\]/g)].map(match => Number(match[1]));
  if (!answer || !references.length || references.some(number => number < 1 || number > evidence.length) ||
      paragraphs.some(part => !/\[\d+\]/.test(part))) return { error: 'invalid_citations', message: 'Local AI returned missing or invalid source references; its answer was withheld.' };
  if (!Array.isArray(parsed.support) || !parsed.support.length || parsed.support.length > 96) {
    return { error: 'invalid_support', message: 'Local AI omitted the source quotations supporting its answer; its answer was withheld.' };
  }
  const support = [];
  for (const item of parsed.support) {
    if (!item || !Number.isSafeInteger(item.paragraph) || item.paragraph < 1 || item.paragraph > paragraphs.length ||
      !Number.isSafeInteger(item.citation) || item.citation < 1 || item.citation > evidence.length ||
      typeof item.quote !== 'string' || item.quote.trim().length < 12 || item.quote.length > 2000 ||
      !paragraphs[item.paragraph - 1].includes(`[${item.citation}]`)) {
      return { error: 'invalid_support', message: 'Local AI returned an invalid paragraph support reference; its answer was withheld.' };
    }
    const source = evidence[item.citation - 1];
    const offset = source.text.indexOf(item.quote);
    if (offset < 0) return { error: 'invalid_support', message: 'A supporting quotation does not occur in the cited source passage; its answer was withheld.' };
    support.push({ ...item, bookId: source.bookId, sectionId: source.sectionId,
      locator: { ...source.locator, charStart: source.locator.charStart + offset,
        charEnd: source.locator.charStart + offset + item.quote.length } });
  }
  if (paragraphs.some((paragraph, index) => [...paragraph.matchAll(/\[(\d+)\]/g)].some(match =>
    !support.some(item => item.paragraph === index + 1 && item.citation === Number(match[1]))))) {
    return { error: 'invalid_support', message: 'Each cited paragraph must have exact support from each referenced source; its answer was withheld.' };
  }
  return { answer, references: [...new Set(references)], support, semanticSupport: 'not_automatically_verified' };
}
