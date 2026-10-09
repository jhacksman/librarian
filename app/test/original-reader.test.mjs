// Execute only through the approved Spark Node runtime.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {renameSync,writeFileSync} from 'node:fs';
import {once} from 'node:events';
import {mkdtemp,open,writeFile,rm,symlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {test} from 'node:test';
import {Writable} from 'node:stream';
import {crc32,deflateRawSync} from 'node:zlib';
import {createOriginalReader,openOriginalArchive,originalPdfRange,resolveOriginalReference,safeOriginalMember,sanitizeOriginalCSS,sanitizeOriginalMarkup} from '../src/original-reader.mjs';
import {pdfOriginalTextMap,originalQuoteSelection,createOriginalView,pdfViewLayout,paintPdfCitation} from '../public/reader-views.mjs';

function zip(entries) {
  const records=[];const central=[];let offset=0;
  for(const entry of entries) {
    const name=Buffer.from(entry.name);const raw=Buffer.from(entry.data||'');const method=entry.method??0;const data=method===8?deflateRawSync(raw):raw;
    const crc=entry.badCrc?(crc32(raw)^1)>>>0:crc32(raw);const local=Buffer.alloc(30);const directory=Buffer.alloc(46);
    local.writeUInt32LE(0x04034b50);local.writeUInt16LE(20,4);local.writeUInt16LE(0x800,6);local.writeUInt16LE(method,8);local.writeUInt32LE(crc,14);
    local.writeUInt32LE(data.length,18);local.writeUInt32LE(raw.length,22);local.writeUInt16LE(name.length,26);
    directory.writeUInt32LE(0x02014b50);directory.writeUInt16LE((3<<8)|20,4);directory.writeUInt16LE(20,6);directory.writeUInt16LE(0x800|(entry.flags||0),8);
    directory.writeUInt16LE(method,10);directory.writeUInt32LE(crc,16);directory.writeUInt32LE(data.length,20);directory.writeUInt32LE(raw.length,24);
    directory.writeUInt16LE(name.length,28);directory.writeUInt32LE(((entry.mode??0o100644)<<16)>>>0,38);directory.writeUInt32LE(offset,42);
    records.push(local,name,data);central.push(directory,name);offset+=30+name.length+data.length;
  }
  const directory=Buffer.concat(central);const end=Buffer.alloc(22);end.writeUInt32LE(0x06054b50);end.writeUInt16LE(entries.length,8);end.writeUInt16LE(entries.length,10);
  end.writeUInt32LE(directory.length,12);end.writeUInt32LE(offset,16);return Buffer.concat([...records,directory,end]);
}
async function directory(t) {const root=await mkdtemp(path.join(tmpdir(),'original-reader-'));t.after(()=>rm(root,{recursive:true,force:true}));return root;}
const resources=member=>`/api/sections/s/original/resource?member=${encodeURIComponent(member)}`;
const sections=member=>({'OPS/chapter.xhtml':'/api/sections/s/original/document','OPS/second.xhtml':'/api/sections/second/original/document'}[member]||null);

test('PDF source item mapping accounts for whitespace, lines and exact UTF-16 citation boundaries',()=> {
  const item=(str,x,y,width=20,hasEOL=false)=>({str,transform:[10,0,0,10,x,y],height:10,width,hasEOL});
  const result=pdfOriginalTextMap([item('Alpha',10,100),item('beta',33,100),item('Second 😀',10,80,50,true)]);
  assert.equal(result.text,'Alpha beta\nSecond 😀');assert.equal(result.items[result.text.indexOf('beta')],1);
  assert.equal(result.items[result.text.indexOf('Second')],2);assert.deepEqual(originalQuoteSelection(result.text,{charStart:6,charEnd:10}),{start:6,end:10,text:'beta'});
  assert.equal(originalQuoteSelection(result.text,{charStart:-1,charEnd:10}),null);assert.equal(originalQuoteSelection(result.text,{charStart:1,charEnd:100}),null);
});

test('original member references stay within archive and preserve decoded anchors',()=> {
  assert.deepEqual(resolveOriginalReference('OPS/chapter.xhtml','../images/plate.png#caption%20one'),{member:'images/plate.png',fragment:'caption one'});
  for(const value of ['../escape','/escape','C:/escape','a\\b','a/./b','a//b','a\0b'])assert.throws(()=>safeOriginalMember(value));
  for(const value of ['../../escape','https://remote.test/x','//remote.test/x','%2fetc/passwd','%5cwindows','file:x','x?query'])assert.throws(()=>resolveOriginalReference('OPS/chapter.xhtml',value));
});

test('formatted EPUB preserves local images/styles and chapter anchors while removing active or external content',async()=> {
  const output=await sanitizeOriginalMarkup(Buffer.from('<html><head><link rel="stylesheet" href="style.css"/><style>p{color:red;background:url(../images/plate.png)}</style></head><body onload="attack()"><h1 id="opening">Chapter</h1><p class="body" style="font-weight:bold">Original <em>emphasis</em></p><img src="../images/plate.png" onerror="attack()"/><a href="second.xhtml#note">Next</a><a href="https://remote.test">Remote</a><script>attack()</script><iframe src="/api/books"></iframe><form action="/api/imports"></form></body></html>'),{member:'OPS/chapter.xhtml',resourceURL:resources,sectionURL:sections});
  assert.match(output,/<h1 id="opening">Chapter/);assert.match(output,/<em>emphasis<\/em>/);assert.match(output,/rel="stylesheet" href="\/api\/sections\/s\/original\/resource/);
  assert.match(output,/member=images%2Fplate.png/);assert.match(output,/href="\/api\/sections\/second\/original\/document#note" data-reader-fragment="note"/);
  assert.doesNotMatch(output,/attack|onload|onerror|<script|<iframe|<form|https:\/\//);
});

test('CSS rejects remote imports, escaped URLs and executable styles',()=> {
  const output=sanitizeOriginalCSS('@import "https://remote.test/css"; p{color:red;background:url(../images/plate.png)} h1{background:url(https://remote.test/track)}','OPS/style.css',resources);
  assert.match(output,/color:red/);assert.match(output,/member=images%2Fplate.png/);assert.doesNotMatch(output,/https|@import/);
  assert.equal(sanitizeOriginalCSS('p{background:u\\72l(https://remote.test)}','OPS/style.css',resources),'');
  assert.equal(sanitizeOriginalCSS('p{width:expression(attack())}','OPS/style.css',resources),'');
});

test('SVG keeps static geometry and drops scripts, event handlers and external references',async()=> {
  const output=await sanitizeOriginalMarkup(Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><rect width="20" height="20" fill="red" onclick="attack()"/><script>attack()</script><foreignObject><iframe src="https://remote.test"/></foreignObject><use href="https://remote.test/x"/></svg>'),{member:'images/plate.svg',resourceURL:resources,sectionURL:sections,svgOnly:true});
  assert.match(output,/viewBox="0 0 20 20"/);assert.match(output,/fill="red"/);assert.doesNotMatch(output,/attack|foreign|iframe|https|onclick/);
});

test('EPUB legacy named anchors and XML IDs survive in original markup',async()=> {
  const output=await sanitizeOriginalMarkup(Buffer.from('<body><h2 xml:id="xml-heading">Heading</h2><a name="legacy-note"></a><p id="modern-note" xml:id="other-note">Note</p></body>'),{member:'OPS/chapter.xhtml',resourceURL:resources,sectionURL:sections});
  assert.match(output,/xml:id="xml-heading" id="xml-heading"/);assert.match(output,/<a name="legacy-note">/);
  assert.match(output,/id="modern-note" xml:id="other-note"/);
});

test('namespace-prefixed XHTML and SVG retain safe original content',async()=> {
  const output=await sanitizeOriginalMarkup(Buffer.from('<x:html xmlns:x="http://www.w3.org/1999/xhtml"><x:body><x:h1 xml:id="prefixed">Heading</x:h1><x:p>Formatted <x:em>text</x:em></x:p><s:svg xmlns:s="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><s:rect width="20" height="20" fill="red"/><s:script>attack()</s:script></s:svg></x:body></x:html>'),{member:'OPS/chapter.xhtml',resourceURL:resources,sectionURL:sections});
  assert.match(output,/<h1 xml:id="prefixed" id="prefixed">Heading/);assert.match(output,/<em>text<\/em>/);assert.match(output,/<svg[^>]*xmlns="http:\/\/www.w3.org\/2000\/svg"/);assert.doesNotMatch(output,/attack|script/);
});

test('legacy passive formatting and SVG local raster plates preserve source content',async()=> {
  const output=await sanitizeOriginalMarkup(Buffer.from('<body><center><font face="serif" color="#223344">Original centered prose</font></center><svg viewBox="0 0 200 300"><image x="0" y="0" width="200" height="300" xlink:href="../images/plate.png"/><image href="https://remote.test/track.png"/></svg></body>'),{member:'OPS/chapter.xhtml',resourceURL:resources,sectionURL:sections});
  assert.match(output,/<center><font color="[^\"]+" face="serif">Original centered prose/);
  assert.match(output,/<image[^>]*href="\/api\/sections\/s\/original\/resource\?member=images%2Fplate.png"/);assert.doesNotMatch(output,/https|track.png/);
});

test('original archive verifies CRC and rejects duplicate paths, links, encrypted data and expansion bombs',async t=> {
  const root=await directory(t);let index=0;
  for(const entries of [[{name:'OPS/a.xhtml',data:'hello',badCrc:true}], [{name:'same',data:'a'},{name:'same',data:'b'}],
    [{name:'link',data:'target',mode:0o120777}],[{name:'encrypted',data:'x',flags:1}],[{name:'../../escape',data:'x'}],
    [{name:'bomb',data:'a'.repeat(2000000),method:8}]]) {
    const filename=path.join(root,`case-${index++}.epub`);await writeFile(filename,zip(entries));
    await assert.rejects(async()=>{const archive=await openOriginalArchive(filename);try{await archive.read(entries[0].name);}finally{archive.close();}});
  }
  const filename=path.join(root,'valid.epub');await writeFile(filename,zip([{name:'OPS/a.xhtml',data:'valid'}]));
  const archive=await openOriginalArchive(filename);try{assert.equal((await archive.read('OPS/a.xhtml')).toString(),'valid');await assert.rejects(()=>archive.read('OPS/a.xhtml',2));}finally{archive.close();}
});

function fakeStore(section,file,others=[]) {return {db:{prepare(sql) {return {get(...params) {
  if(sql.includes('FROM sections'))return params[0]===section.id?section:undefined;
  if(sql.includes('FROM files'))return params[0]===file.book_id&&params[1]===file.sha256&&params[2]===file.format&&file.staged===1?file:undefined;
},all(){return [section,...others];}};}}};}

test('EPUB member teardown and read failures leave the verified descriptor owned by its caller',async t=> {
  const root=await directory(t);const text='Retained original member. '.repeat(5000);
  const bytes=zip([{name:'OPS/a.xhtml',data:text,method:8},{name:'OPS/b.xhtml',data:'Another original member'},
    {name:'OPS/bad.xhtml',data:'Rejected member checksum',badCrc:true}]);
  const filename=path.join(root,'owned.epub');await writeFile(filename,bytes);
  const handle=await open(filename,'r');const owned={handle,size:bytes.length};
  let archive;
  try {
    archive=await openOriginalArchive(owned);
    for(let count=0;count<3;count++)assert.equal((await archive.read('OPS/a.xhtml')).toString(),text);
    await assert.rejects(()=>archive.read('OPS/bad.xhtml'),/CRC/);
    assert.equal((await handle.stat()).size,bytes.length);
    assert.equal((await archive.read('OPS/b.xhtml')).toString(),'Another original member');
    archive.close();archive=null;
    assert.equal((await handle.stat()).size,bytes.length);
    const limits={entries:10,member:200000,total:text.length+100,ratio:1000};
    archive=await openOriginalArchive(owned,limits);
    assert.equal((await archive.read('OPS/a.xhtml')).toString(),text);
    // A repeated read exceeds the cumulative bound after the stream starts.
    await assert.rejects(()=>archive.read('OPS/a.xhtml'),/expansion limit/);
    archive.close();archive=null;
    const prefix=Buffer.alloc(2);assert.equal((await handle.read(prefix,0,2,0)).bytesRead,2);
    assert.equal(prefix.toString(),'PK');
  } finally {archive?.close();await handle.close();}
});

function response() {return {headers:{},setHeader(k,v){this.headers[k]=v;},writeHead(code){this.status=code;},end(body){this.body=body;}};}
async function fixture(t,format='epub') {
  const root=await directory(t);const bytes=format==='pdf'?Buffer.from('%PDF-1.7 source fixture'):zip([{name:'OPS/chapter.xhtml',data:'<h1 id="opening">Formatted chapter</h1><img src="plate.png"/><a href="second.xhtml#note">Next</a>'},{name:'OPS/plate.png',data:'image fixture'},{name:'OPS/second.xhtml',data:'<h2 id="note">Second</h2>'}]);
  const sha=createHash('sha256').update(bytes).digest('hex');const filename=path.join(root,`source.${format}`);await writeFile(filename,bytes);
  const section={id:'s',book_id:sha,title:'Chapter',locator_json:JSON.stringify(format==='pdf'?{format,page:7}:{format,member:'OPS/chapter.xhtml'})};
  const file={id:'f',book_id:sha,sha256:sha,format,staged:1,path:filename,relative_path:`source.${format}`};return {root,bytes,section,file};
}

test('metadata and EPUB document remain attached to exact source book; EPUB has no invented page number',async t=> {
  const f=await fixture(t);const other={id:'second',book_id:f.section.book_id,locator_json:JSON.stringify({format:'epub',member:'OPS/second.xhtml'})};
  const handler=createOriginalReader({store:fakeStore(f.section,f.file,[other]),dataDir:f.root,sendManagedFile:()=>assert.fail('EPUB should not stream raw source')});
  const metadata=response();assert.equal(await handler.handle({method:'GET'},metadata,new URL('http://local/api/sections/s/original')),true);
  const payload=JSON.parse(metadata.body);assert.equal(payload.bookId,f.file.sha256);assert.equal(payload.member,'OPS/chapter.xhtml');assert.equal(payload.reflowable,true);assert.equal(payload.physicalPage,undefined);
  const doc=response();await handler.handle({method:'GET'},doc,new URL('http://local/api/sections/s/original/document'));
  assert.match(doc.body,/Formatted chapter/);assert.match(doc.body,/\/second\/original\/document#note/);assert.match(doc.headers['Content-Security-Policy'],/script-src 'none'/);
  assert.match(doc.headers['Content-Security-Policy'],/sandbox allow-same-origin/);
  await assert.rejects(()=>handler.handle({method:'GET'},response(),new URL('http://local/api/sections/s/original/resource?member=..%2Foutside')));
  await assert.rejects(()=>handler.handle({method:'GET'},response(),new URL('http://local/api/sections/s/original/resource?member=OPS%2Fchapter.xhtml')));
});

test('PDF descriptor uses physical locator page and verified PDF route',async t=> {
  const f=await fixture(t,'pdf');
  const handler=createOriginalReader({store:fakeStore(f.section,f.file),dataDir:f.root});
  const res=response();await handler.handle({method:'GET'},res,new URL('http://local/api/sections/s/original'));assert.equal(JSON.parse(res.body).physicalPage,7);
  const pdf=response();await handler.handle({method:'HEAD'},pdf,new URL('http://local/api/sections/s/original/pdf'));assert.equal(pdf.headers['Content-Type'],'application/pdf');assert.equal(pdf.headers['Content-Length'],f.bytes.length);
  const ranged=response();await handler.handle({method:'HEAD',headers:{range:'bytes=0-4'}},ranged,new URL('http://local/api/sections/s/original/pdf'));assert.equal(ranged.status,206);assert.equal(ranged.headers['Content-Length'],5);
  assert.deepEqual(originalPdfRange('bytes=-5',20),{start:15,end:19,status:206});
  for(const range of ['bytes=-0','bytes=20-','bytes=9-2','bytes=0-1,5-6','bytes=9007199254740993-'])assert.throws(()=>originalPdfRange(range,20));
});

test('PDF streams the verified descriptor even when its source path is replaced after verification',async t=> {
  const f=await fixture(t,'pdf');const chunks=[];
  const res=new Writable({write(chunk,_encoding,callback){chunks.push(Buffer.from(chunk));callback();}});
  res.headers={};res.setHeader=(name,value)=>{res.headers[name]=value;};
  res.writeHead=()=>{renameSync(f.file.path,`${f.file.path}.retained`);writeFileSync(f.file.path,'replacement from another book');};
  const finished=once(res,'finish');
  const handler=createOriginalReader({store:fakeStore(f.section,f.file),dataDir:f.root});
  await handler.handle({method:'GET'},res,new URL('http://local/api/sections/s/original/pdf'));await finished;
  assert.deepEqual(Buffer.concat(chunks),f.bytes);
});

test('source identity refuses another book, changed bytes, external path, symlink, derived format and malformed PDF page',async t=> {
  const f=await fixture(t);const request=()=>new URL('http://local/api/sections/s/original');
  const invoke=()=>createOriginalReader({store:fakeStore(f.section,f.file),dataDir:f.root,sendManagedFile:()=>{}}).handle({method:'GET'},response(),request());
  const originalBook=f.section.book_id;f.section.book_id='0'.repeat(64);await assert.rejects(invoke,/Verified original source/);f.section.book_id=originalBook;
  await writeFile(f.file.path,'changed');await assert.rejects(invoke,/identity check/);await writeFile(f.file.path,f.bytes);
  const outside=await directory(t);const external=path.join(outside,'source.epub');await writeFile(external,f.bytes);const originalPath=f.file.path;f.file.path=external;await assert.rejects(invoke,/outside this library/);
  const link=path.join(f.root,'link.epub');await symlink(external,link);f.file.path=link;await assert.rejects(invoke,/rendering limits/);f.file.path=originalPath;
  f.section.locator_json=JSON.stringify({format:'epub',member:'OPS/chapter.xhtml',derived:{format:'epub'}});await assert.rejects(invoke,/unavailable/);
  const pdf=await fixture(t,'pdf');pdf.section.locator_json=JSON.stringify({format:'pdf',page:0});await assert.rejects(()=>createOriginalReader({store:fakeStore(pdf.section,pdf.file),dataDir:pdf.root,sendManagedFile:()=>{}}).handle({method:'GET'},response(),request()),/physical page/);
});


test('closing an EPUB view with a never-loading iframe settles readiness', {timeout:1000}, async t=> {
  class Element extends EventTarget {
    constructor(tag){super();this.tagName=tag.toUpperCase();this.dataset={};this.children=[];this.isConnected=true;this.clientWidth=400;this.listeners=new Map();}
    addEventListener(type,callback,options){if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(callback);super.addEventListener(type,callback,options);}
    removeEventListener(type,callback,options){this.listeners.get(type)?.delete(callback);super.removeEventListener(type,callback,options);}
    setAttribute(){} append(...children){this.children.push(...children);} replaceChildren(...children){this.children=children;}
  }
  const savedDocument=globalThis.document,savedFetch=globalThis.fetch;
  const elements=[];
  globalThis.document={head:{append(){}},createElement(tag){const element=new Element(tag);elements.push(element);return element;}};
  globalThis.fetch=async()=>({ok:true,json:async()=>({sectionId:'section',bookId:'book',sourceSha256:'book',format:'epub',documentURL:'/never-load'})});
  t.after(()=>{globalThis.document=savedDocument;globalThis.fetch=savedFetch;});
  const view=await createOriginalView({section:{id:'section',bookId:'book',text:'Source'},book:{id:'book'},position:{scrollTop:217,scrollLeft:9}});
  for(let turn=0;turn<3;turn++)await Promise.resolve();
  const frame=elements.find(element=>element.tagName==='IFRAME');
  assert.ok(frame, 'The original iframe exists and its load remains pending');
  assert.equal(frame.listeners.get('load').size,1);assert.equal(frame.listeners.get('error').size,1);
  frame.contentDocument={scrollingElement:{scrollTop:0,scrollLeft:0}};
  assert.equal(view.capturePosition().scrollTop,217);assert.equal(view.capturePosition().scrollLeft,9);
  const closed=assert.rejects(view.ready,{name:'AbortError'});
  view.destroy();await closed;
  assert.equal(frame.listeners.get('load').size,0);assert.equal(frame.listeners.get('error').size,0);
});

test('PDF readable size permits mobile panning without exceeding the raster budget',()=> {
  const natural={width:612,height:792};
  const fit=pdfViewLayout(natural,278,1,'fit');assert.equal(fit.cssWidth,278);
  const readable=pdfViewLayout(natural,278,2,'readable');assert.equal(readable.cssWidth,640);
  const enlarged=pdfViewLayout(natural,278,2,'200');assert.equal(enlarged.cssWidth,1280);
  for(const dimensions of [natural,{width:612,height:20000}]) {
    const layout=pdfViewLayout(dimensions,278,4,'200');
    const width=Math.floor(dimensions.width*layout.renderScale),height=Math.floor(dimensions.height*layout.renderScale);
    assert.ok(width*height<=16000000);
  }
  assert.throws(()=>pdfViewLayout({width:0,height:792},278,1),/dimensions/);
});

test('PDF citation overlay paints through real installed viewport transforms at every rotation', {timeout:5000}, async()=> {
  const objects=['<< /Type /Catalog /Pages 2 0 R >>','<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>',
    '<< /Length 0 >>\nstream\nendstream'];
  let source='%PDF-1.4\n';const offsets=[0];
  for(const [index,object] of objects.entries()){offsets.push(Buffer.byteLength(source));source+=`${index+1} 0 obj\n${object}\nendobj\n`;}
  const xref=Buffer.byteLength(source);
  source+=`xref\n0 5\n0000000000 65535 f \n${offsets.slice(1).map(offset=>String(offset).padStart(10,'0')+' 00000 n \n').join('')}trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  const {getDocument}=await import('pdfjs-dist/legacy/build/pdf.mjs');
  const {createCanvas}=await import('@napi-rs/canvas');
  const task=getDocument({data:Uint8Array.from(Buffer.from(source)),verbosity:0,stopAtErrors:true,
    useSystemFonts:false,disableFontFace:true,enableXfa:false,useWasm:false});
  try {
    const document=await task.promise;const page=await document.getPage(1);
    const item={transform:[10,0,0,10,10,40],height:10,width:20};const original=structuredClone(item);
    // PDF corners (10,38) and (30,48.5), transformed onto a 200x200 page.
    const expected=[[0,20,103,40,21],[90,76,20,21,40],[180,140,76,40,21],[270,103,140,21,40]];
    for(const [rotation,left,top,width,height] of expected) {
      const viewport=page.getViewport({scale:2,rotation});const canvas=createCanvas(viewport.width,viewport.height);const ctx=canvas.getContext('2d');
      ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);
      const pixel=(x,y)=>[...ctx.getImageData(x,y,1,1).data];
      const before=ctx.getImageData(0,0,canvas.width,canvas.height).data;
      paintPdfCitation(ctx,viewport,[item],new Set());
      assert.deepEqual(ctx.getImageData(0,0,canvas.width,canvas.height).data,before);
      paintPdfCitation(ctx,viewport,[item],new Set([0]));
      const painted=pixel(left+2,top+2);assert.equal(painted[0],255);assert.ok(painted[1]<255&&painted[2]<painted[1]);assert.equal(painted[3],255);
      for(const [x,y] of [[left-1,top+2],[left+width,top+2],[left+2,top-1],[left+2,top+height]])assert.deepEqual(pixel(x,y),[255,255,255,255]);
      assert.equal(ctx.globalCompositeOperation,'source-over');assert.deepEqual(item,original);
      // @napi-rs/canvas v1.0.10 caches the last fillStyle assignment separately
      // from saved native paint state. Verify restoration through actual paint.
      ctx.fillRect(left,top,width,height);
      assert.deepEqual(pixel(left+2,top+2),[255,255,255,255]);
    }
  } finally {await task.destroy();}
});
