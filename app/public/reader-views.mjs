// Original views use server-verified source bytes. The application owns the
// extracted-text toggle and persists each view's position independently.
const node = (tag, props = {}) => Object.assign(document.createElement(tag), props);
let styles;
function loadStyles() {
  if (!styles) {styles=node('link',{rel:'stylesheet',href:'/reader-views.css'});document.head.append(styles);}
}
function finite(value, fallback = 0) {return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : fallback;}

// Reproduce the extractor's whitespace decisions, retaining the source item
// behind every character. Only exact source-text equality authorizes a region.
export function pdfOriginalTextMap(items) {
  const parts=[];const owners=[];let previous;
  const append=(text,owner=-1)=>{parts.push(text);for(let index=0;index<text.length;index++)owners.push(owner);};
  items.forEach((item,index)=> {
    if(typeof item.str!=='string')return;
    const transform=item.transform??[1,0,0,1,0,0];const x=transform[4];const y=transform[5];const height=Math.max(1,Math.abs(item.height||transform[3]||10));
    if(previous&&item.str) {
      const gap=x-previous.end;
      if(Math.abs(y-previous.y)>Math.max(2,height*0.4)){if(!previous.eol)append('\n');}
      else if(!previous.eol&&!/\s$/.test(previous.text)&&!/^\s/.test(item.str)) {
        if(gap>height*2)append('\t');else if(gap>height*0.1)append(' ');
      }
    }
    append(item.str,index);if(item.hasEOL)append('\n');previous={y,end:x+(item.width||0),text:item.str,eol:item.hasEOL};
  });
  const raw=parts.join('');let text='';const map=[];let newlines=0;
  for(let index=0;index<raw.length;index++) {
    const char=raw[index];if(char==='\0')continue;
    newlines=char==='\n'?newlines+1:0;if(newlines>2)continue;
    text+=char;map.push(owners[index]);
  }
  const leading=text.length-text.trimStart().length;const result=text.trim();
  return {text:result,items:map.slice(leading,leading+result.length)};
}

export function originalQuoteSelection(sourceText, position) {
  const start=position?.charStart;const end=position?.charEnd;
  return Number.isSafeInteger(start)&&Number.isSafeInteger(end)&&start>=0&&end>start&&end<=sourceText.length?{start,end,text:sourceText.slice(start,end)}:null;
}

// Closing a view must settle readiness even if a detached iframe never loads.
export function waitForOriginalView(task, signal) {
  return new Promise((resolve,reject)=> {
    const closed=()=>{signal.removeEventListener('abort',closed);reject(new DOMException('Original view closed','AbortError'));};
    Promise.resolve(task).then(value=>{signal.removeEventListener('abort',closed);resolve(value);},cause=>{signal.removeEventListener('abort',closed);reject(cause);});
    if(signal.aborted){closed();return;}
    signal.addEventListener('abort',closed,{once:true});
  });
}

export function pdfViewLayout(natural, availableWidth, pixelRatio, zoom = 'fit') {
  if(!Number.isFinite(natural?.width)||natural.width<=0||!Number.isFinite(natural?.height)||natural.height<=0)throw new Error('Invalid original PDF dimensions.');
  const fit=Math.max(1,finite(availableWidth,1));
  const factor=zoom==='150'?1.5:zoom==='200'?2:1;
  const cssWidth=zoom==='fit'?fit:Math.max(640,fit)*factor;
  const renderScale=Math.min(cssWidth/natural.width*Math.max(1,finite(pixelRatio,1)),Math.sqrt(16000000/(natural.width*natural.height)));
  return {cssWidth,renderScale};
}

export function paintPdfCitation(ctx, viewport, items, selected) {
  ctx.save();ctx.globalCompositeOperation='multiply';ctx.fillStyle='rgba(255,218,74,0.45)';
  for(const index of selected) {
    const item=items[index];const x=item.transform[4];const y=item.transform[5];const height=Math.max(1,Math.abs(item.height||item.transform[3]||10));
    const [left,top]=viewport.convertToViewportPoint(x,y-height*0.2);
    const [right,bottom]=viewport.convertToViewportPoint(x+item.width,y+height*0.85);
    ctx.fillRect(Math.min(left,right),Math.min(top,bottom),Math.abs(right-left),Math.abs(bottom-top));
  }
  ctx.restore();
}

export async function createOriginalView({section,book,position = {},onNavigate = () => {},onPosition = () => {},signal}) {
  if (!section?.id || !section.bookId || String(section.bookId)!==String(book?.id)) throw new Error('Original view belongs to another book.');
  loadStyles();
  const response=await fetch(`/api/sections/${encodeURIComponent(section.id)}/original`,{signal});
  const metadata=await response.json();
  if(!response.ok)throw new Error(metadata.error || 'Original source is unavailable.');
  if(metadata.sectionId!==section.id || metadata.bookId!==section.bookId || metadata.sourceSha256!==section.bookId)throw new Error('Original source identity mismatch.');
  const element=node('div',{className:`original-reader original-reader-${metadata.format}`});
  element.dataset.testid='original-reader';element.dataset.bookId=section.bookId;element.dataset.sectionId=section.id;
  const status=node('p',{className:'original-reader-status',textContent:'Loading original source…'});status.setAttribute('role','status');element.append(status);
  let disposed=false;let nativePositionReady=false;let documentTask;let renderTask;let frame;let pdfViewport;let observer;let cleanup=()=>{};
  const lifetime=new AbortController();
  let current={...position,format:metadata.format,sectionId:section.id};
  const quote=originalQuoteSelection(String(section.text||''),position);
  const quoteStatus=node('p',{className:'original-quote-status'});quoteStatus.dataset.testid='original-quote-status';
  function quoteMessage(mapped) {if(!quote)return;quoteStatus.textContent=mapped?'Highlighted source region contains the cited passage.':'The cited passage is on this original page or section. Choose Extracted text to see its exact marked wording.';element.append(quoteStatus);}
  const notify=()=>{if(!disposed)onPosition(capturePosition());};
  function capturePosition() {
    if(nativePositionReady&&frame?.contentDocument?.scrollingElement) {
      const scroll=frame.contentDocument.scrollingElement;current={...current,member:metadata.member,scrollTop:scroll.scrollTop,scrollLeft:scroll.scrollLeft};
    } else if(nativePositionReady&&metadata.format==='pdf')current={...current,page:metadata.physicalPage,scrollTop:pdfViewport?.scrollTop||0,scrollLeft:pdfViewport?.scrollLeft||0};
    return {...current};
  }
  function restorePosition(value={}) {
    current={...current,...value};
    const scroll=frame?.contentDocument?.scrollingElement || pdfViewport || element;
    if(frame && value.fragment && value.scrollTop===undefined) {
      const doc=frame.contentDocument;
      const anchor=doc?.getElementById(value.fragment)||Array.from(doc?.getElementsByName(value.fragment)||[]).find(item=>item.tagName==='A')
        ||Array.from(doc?.querySelectorAll('[xml\\:id]')||[]).find(item=>item.getAttribute('xml:id')===value.fragment);
      if(anchor){anchor.scrollIntoView();return;}
    }
    scroll.scrollTop=finite(value.scrollTop);scroll.scrollLeft=finite(value.scrollLeft);
  }
  const abort=()=>{disposed=true;lifetime.abort();renderTask?.cancel();documentTask?.destroy();observer?.disconnect();cleanup();};
  signal?.addEventListener('abort',abort,{once:true});
  const mounted = () => new Promise((resolve,reject)=> {
    const tick=()=>{if(disposed||signal?.aborted)return reject(new DOMException('Aborted','AbortError'));if(element.isConnected&&element.clientWidth>0)return resolve();requestAnimationFrame(tick);};tick();
  });
  const task=(async()=> {
    await mounted();
    if(metadata.format==='pdf') {
      const pdfjs=await import('/api/reader-assets/build/pdf.mjs');if(disposed)return;
      pdfjs.GlobalWorkerOptions.workerSrc='/api/reader-assets/build/pdf.worker.mjs';
      documentTask=pdfjs.getDocument({url:metadata.pdfURL,cMapUrl:'/api/reader-assets/cmaps/',cMapPacked:true,
        standardFontDataUrl:'/api/reader-assets/standard_fonts/',wasmUrl:'/api/reader-assets/wasm/',
        isEvalSupported:false,disableFontFace:true,enableXfa:false});
      const pdf=await documentTask.promise;if(disposed)return;
      if(metadata.physicalPage>pdf.numPages)throw new Error('The cited physical PDF page is unavailable.');
      const page=await pdf.getPage(metadata.physicalPage);if(disposed)return;
      const content=quote?await page.getTextContent({disableNormalization:true}):null;if(disposed)return;
      const sourceMap=content?pdfOriginalTextMap(content.items):null;
      const selected=sourceMap?.text===String(section.text||'')?new Set(sourceMap.items.slice(quote.start,quote.end).filter(index=>index>=0)):new Set();
      const canvas=node('canvas',{className:'original-pdf-page'});canvas.dataset.testid='original-pdf-canvas';
      canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`Original PDF physical page ${metadata.physicalPage} of ${pdf.numPages}`);
      const label=node('p',{className:'original-page-label',textContent:`Original PDF · physical page ${metadata.physicalPage} of ${pdf.numPages}`});
      pdfViewport=node('div',{className:'original-pdf-viewport'});pdfViewport.dataset.testid='original-pdf-viewport';
      pdfViewport.setAttribute('tabindex','0');pdfViewport.setAttribute('aria-label','Original PDF page; scroll to read at the selected size');
      let zoom=['fit','readable','150','200'].includes(current.pdfZoom)?current.pdfZoom:window.matchMedia('(max-width:640px)').matches?'readable':'fit';
      const zoomLabel=node('label',{className:'original-pdf-zoom',textContent:'Page size '});
      const zoomControl=node('select');zoomControl.setAttribute('aria-label','PDF page size');zoomControl.dataset.testid='original-pdf-zoom';
      for(const [value,text] of [['fit','Fit width'],['readable','Readable'],['150','150%'],['200','200%']])zoomControl.append(node('option',{value,textContent:text}));
      zoomControl.value=zoom;zoomLabel.append(zoomControl);
      let renderSequence=0;
      const render = async () => {
        const sequence=++renderSequence;
        if(disposed)return false;
        pdfViewport.setAttribute('aria-busy','true');
        if(renderTask){renderTask.cancel();await renderTask.promise.catch(()=>{});}
        if(disposed||sequence!==renderSequence)return false;
        const natural=page.getViewport({scale:1});const layout=pdfViewLayout(natural,element.clientWidth-24,window.devicePixelRatio,zoom);
        const viewport=page.getViewport({scale:layout.renderScale});
        canvas.style.width=`${layout.cssWidth}px`;
        canvas.width=Math.max(1,Math.floor(viewport.width));canvas.height=Math.max(1,Math.floor(viewport.height));
        renderTask=page.render({canvasContext:canvas.getContext('2d',{alpha:false}),viewport});
        try {await renderTask.promise;}catch(cause){if(cause.name==='RenderingCancelledException')return false;throw cause;}
        if(disposed||sequence!==renderSequence)return false;
        if(selected.size) {
          paintPdfCitation(canvas.getContext('2d'),viewport,content.items,selected);
          canvas.dataset.citationHighlighted='true';
        }
        pdfViewport.setAttribute('aria-busy','false');return true;
      };
      await render();if(disposed)return;
      pdfViewport.append(canvas);element.replaceChildren(label,zoomLabel,pdfViewport);quoteMessage(selected.size>0);restorePosition(position);current.pdfZoom=zoom;
      const changeZoom=()=>{zoom=zoomControl.value;current.pdfZoom=zoom;void render().then(rendered=>{if(rendered)notify();}).catch(cause=>{if(!disposed){status.textContent=cause.message;element.append(status);}});};
      zoomControl.addEventListener('change',changeZoom);
      pdfViewport.addEventListener('scroll',notify,{passive:true});
      let width=element.clientWidth;let queued;
      observer=new ResizeObserver(()=>{if(element.clientWidth===width)return;width=element.clientWidth;clearTimeout(queued);queued=setTimeout(()=>render().catch(cause=>{status.textContent=cause.message;element.append(status);}),100);});observer.observe(element);
      cleanup=()=>{pdfViewport.removeEventListener('scroll',notify);zoomControl.removeEventListener('change',changeZoom);clearTimeout(queued);};
    } else if(metadata.format==='epub') {
      frame=node('iframe',{className:'original-epub-frame',title:`Original EPUB section: ${section.title||book.title||'Book'}`});
      frame.dataset.testid='original-epub-frame';frame.setAttribute('sandbox','allow-same-origin');frame.setAttribute('referrerpolicy','no-referrer');
      const loaded=new Promise((resolve,reject)=> {
        const clear=()=>{frame.removeEventListener('load',done);frame.removeEventListener('error',failed);};
        const done=()=>{clear();resolve();};
        const failed=()=>{clear();reject(new Error('Original EPUB section could not be loaded.'));};
        frame.addEventListener('load',done,{once:true});frame.addEventListener('error',failed,{once:true});
        cleanup=()=>{clear();reject(new DOMException('Original view closed','AbortError'));};
      });
      frame.src=metadata.documentURL;element.replaceChildren(node('p',{className:'original-page-label',textContent:'Original EPUB · formatted, reflowable section'}),frame);
      await loaded;if(disposed)return;
      const doc=frame.contentDocument;
      if(!doc?.body)throw new Error('Original EPUB section is unavailable.');
      const click=event=> {
        const link=event.target.closest?.('a');if(!link)return;event.preventDefault();
        const href=link.getAttribute('href');if(!href)return;
        const target=new URL(href,frame.src);const match=/^\/api\/sections\/([^/]+)\/original\/document$/.exec(target.pathname);
        if(target.origin!==location.origin||!match)return;
        const sectionId=decodeURIComponent(match[1]);const fragment=link.dataset.readerFragment||'';
        if(sectionId===section.id){current.fragment=fragment;restorePosition({fragment});notify();}
        else onNavigate({sectionId,fragment});
      };
      doc.addEventListener('click',click);doc.addEventListener('scroll',notify,{passive:true});
      restorePosition(position);
      let mapped=false;
      if(quote) {
        const walker=doc.createTreeWalker(doc.body,NodeFilter.SHOW_TEXT,{acceptNode(text){return text.parentElement?.closest('script,style,head,[hidden]')?NodeFilter.FILTER_REJECT:NodeFilter.FILTER_ACCEPT;}});
        const blockSelector='p,div,section,article,header,footer,h1,h2,h3,h4,h5,h6,li,blockquote,figure,figcaption,pre,tr';
        const nodes=[];let raw='';let previousBlock;
        while(walker.nextNode()) {
          const text=walker.currentNode;const block=text.parentElement?.closest(blockSelector);
          if(previousBlock&&block!==previousBlock)raw+='\n';previousBlock=block;
          nodes.push({node:text,start:raw.length,end:raw.length+text.textContent.length});raw+=text.textContent;
        }
        const normalized=value=>value.replace(/\s+/g,' ').trim();const sought=normalized(quote.text);
        let compact='';const indices=[];let space=false;
        for(let index=0;index<raw.length;index++){const char=raw[index];if(/\s/.test(char)){if(!space){compact+=' ';indices.push(index);}space=true;}else{compact+=char;indices.push(index);space=false;}}
        const offset=compact.indexOf(sought);
        if(sought&&offset>=0&&compact.indexOf(sought,offset+1)<0) {
          const first=indices[offset];const last=indices[offset+sought.length-1]+1;
          const a=nodes.find(item=>first>=item.start&&first<item.end);const b=nodes.find(item=>last>item.start&&last<=item.end);
          if(a&&b) {
            const range=doc.createRange();range.setStart(a.node,first-a.start);range.setEnd(b.node,last-b.start);
            if(frame.contentWindow.CSS?.highlights&&frame.contentWindow.Highlight) {
              const style=doc.createElement('style');style.textContent='::highlight(librarian-citation){background:#ffe17a;color:inherit}';doc.head.append(style);
              frame.contentWindow.CSS.highlights.set('librarian-citation',new frame.contentWindow.Highlight(range));mapped=true;
              if(position.scrollTop===undefined) {const rect=range.getBoundingClientRect();frame.contentWindow.scrollTo(0,Math.max(0,rect.top+frame.contentWindow.scrollY-80));}
            }
          }
        }
      }
      quoteMessage(mapped);
      cleanup=()=>{doc.removeEventListener('click',click);doc.removeEventListener('scroll',notify);};
    } else throw new Error('Original source format is unavailable.');
    nativePositionReady=true;notify();return metadata;
  })();
  const ready=waitForOriginalView(task,lifetime.signal);
  // Failures are visible in the mounted view and remain observable to callers.
  ready.catch(cause=>{if(!disposed){status.textContent=`Original view unavailable: ${cause.message}`;status.setAttribute('role','alert');element.replaceChildren(status);}});
  return {element,ready,metadata,capturePosition,restorePosition,destroy(){capturePosition();abort();signal?.removeEventListener('abort',abort);}};
}
