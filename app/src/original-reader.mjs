import {createHash} from 'node:crypto';
import {constants,createReadStream} from 'node:fs';
import {lstat,open,readFile,realpath} from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {crc32} from 'node:zlib';

const require = createRequire(import.meta.url);
const error = (message, statusCode = 422) => Object.assign(new Error(message), {statusCode});
const check = (condition, message, status = 422) => { if (!condition) throw error(message, status); };
const parse = value => { try { return JSON.parse(value || '{}'); } catch { return {}; } };
const within = (root, file) => { const rel = path.relative(root, file); return rel !== '..' && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel); };
const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const LIMITS = {source: 512 * 1024 ** 2, entries: 10000, member: 32 * 1024 ** 2, total: 512 * 1024 ** 2, ratio: 1000, markup: 8 * 1024 ** 2};
const TYPES = {'.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.gif':'image/gif', '.webp':'image/webp', '.avif':'image/avif', '.svg':'image/svg+xml', '.css':'text/css; charset=utf-8', '.woff':'font/woff', '.woff2':'font/woff2', '.ttf':'font/ttf', '.otf':'font/otf'};

// autoClose:false only disables automatic destruction at EOF. Explicit
// ReadStream.destroy() still invokes fs.close, including yauzl's teardown of
// member streams. These streams borrow the verified handle; only its owner may
// close it. FileHandle.read also keeps pending I/O tracked during owner cleanup.
function verifiedReadStream(handle,start,end) {
  return createReadStream('',{fd:handle.fd,start,end,autoClose:false,fs:{
    read(_fd,buffer,offset,length,position,callback) {
      handle.read(buffer,offset,length,position).then(
        result=>callback(null,result.bytesRead,result.buffer),cause=>callback(cause));
    },
    close(_fd,callback) {setImmediate(callback,null);},
  }});
}

export function safeOriginalMember(name) {
  check(typeof name === 'string' && name.length > 0 && name.length <= 4096 && !/[\u0000-\u001f\\?#]/.test(name)
    && !name.startsWith('/') && !/^[a-z][a-z0-9+.-]*:/i.test(name)
    && !name.split('/').some(p => p === '.' || p === '..' || p === ''), 'Unsafe original member path');
  return name;
}

export function resolveOriginalReference(base, reference) {
  check(typeof reference === 'string' && reference.length <= 4096 && !/[\u0000-\u0020\\]/.test(reference), 'Unsafe original reference');
  const at = reference.indexOf('#');
  const raw = at < 0 ? reference : reference.slice(0, at);
  const fragment = at < 0 ? '' : decodeURIComponent(reference.slice(at + 1));
  const decoded = decodeURIComponent(raw);
  check(!decoded.startsWith('/') && !/^[a-z][a-z0-9+.-]*:/i.test(decoded) && !/[\u0000-\u001f\\?]/.test(decoded), 'External original reference refused');
  check(!/[\u0000-\u001f]/.test(fragment), 'Unsafe original fragment');
  return {member: safeOriginalMember(raw ? path.posix.normalize(path.posix.join(path.posix.dirname(base), decoded)) : base), fragment};
}

// The reader never unpacks an archive to disk. Every requested member is bounded
// and verified independently; all inventory paths are checked before use.
export async function openOriginalArchive(filename, limits = LIMITS) {
  const {default: yauzl} = await import('yauzl');
  const options={autoClose:false, strictFileNames:true, validateEntrySizes:true};
  let archive;
  if(typeof filename==='string')archive=await yauzl.openPromise(filename,options);
  else {
    // Retain the same verified file descriptor throughout rendering. This
    // adapter deliberately leaves descriptor ownership with the source caller.
    class ManagedReader extends yauzl.RandomAccessReader {
      _readStreamForRange(start,end) {return verifiedReadStream(filename.handle,start,end-1);}
      close(callback) {setImmediate(callback);}
    }
    archive=await yauzl.fromRandomAccessReaderPromise(new ManagedReader(),filename.size,options);
  }
  const entries = new Map(); let total = 0; let inflated = 0;
  try {
    check(archive.entryCount <= limits.entries, 'Too many original archive members');
    for await (const entry of archive.eachEntry()) {
      const directory = entry.fileName.endsWith('/');
      safeOriginalMember(directory ? entry.fileName.slice(0, -1) : entry.fileName);
      check(!entries.has(entry.fileName) && entries.size < limits.entries, 'Duplicate original archive member');
      const type = (entry.externalFileAttributes >>> 16) & 0xf000;
      check([0, 0x8000, 0x4000].includes(type), 'Archive links and special files refused');
      check(!(entry.generalPurposeBitFlag & 1) && [0,8].includes(entry.compressionMethod), 'Encrypted or unsupported archive member');
      check(Number.isSafeInteger(entry.uncompressedSize) && Number.isSafeInteger(entry.compressedSize)
        && entry.uncompressedSize >= 0 && entry.compressedSize >= 0, 'Invalid original archive size');
      check(entry.uncompressedSize <= limits.member && entry.uncompressedSize <= Math.max(1,entry.compressedSize) * limits.ratio, 'Original archive expansion limit exceeded');
      total += entry.uncompressedSize;
      check(total <= limits.total, 'Original archive total expansion limit exceeded');
      entries.set(entry.fileName, entry);
    }
    return {close: () => archive.close(), async read(member, cap = limits.member) {
      const entry = entries.get(safeOriginalMember(member));
      check(entry && !member.endsWith('/'), 'Original member not found', 404);
      check(entry.uncompressedSize <= cap, 'Original member exceeds rendering limit');
      const stream = await archive.openReadStreamPromise(entry); const chunks = []; let size = 0; let checksum = 0;
      try { for await (const chunk of stream) {
        size += chunk.length; inflated += chunk.length;
        check(size <= cap && inflated <= limits.total, 'Original member expansion limit exceeded');
        checksum = crc32(chunk, checksum); chunks.push(chunk);
      }} finally { stream.destroy(); }
      check(size === entry.uncompressedSize && checksum === entry.crc32, 'Original member CRC or size mismatch');
      return Buffer.concat(chunks, size);
    }};
  } catch (cause) { archive.close(); throw cause; }
}

function decodeMarkup(bytes) {
  const encoding = (bytes[0] === 0xff && bytes[1] === 0xfe) || (bytes[0] === 0x3c && bytes[1] === 0) ? 'utf-16le'
    : (bytes[0] === 0xfe && bytes[1] === 0xff) || (bytes[0] === 0 && bytes[1] === 0x3c) ? 'utf-16be' : 'utf-8';
  return new TextDecoder(encoding, {fatal:true}).decode(bytes);
}

export function sanitizeOriginalCSS(css, base, resourceURL) {
  check(Buffer.byteLength(css) <= LIMITS.markup, 'Original stylesheet exceeds rendering limit');
  // Escaped URLs and legacy execution syntax have no role in this preview.
  css = css.replace(/\/\*[\s\S]*?\*\//g, '');
  if (/[\\\u0000]|expression\s*\(|-moz-binding|behavior\s*:|<\/style/i.test(css)) return '';
  css = css.replace(/@import\b[^;{}]*(?:;|$)/gi, '');
  return css.replace(/url\(\s*(?:"([^"\n]*)"|'([^'\n]*)'|([^)'"\n]*))\s*\)/gi, (_all,a,b,c) => {
    try { const target = resolveOriginalReference(base, (a ?? b ?? c).trim());
      if (!TYPES[path.posix.extname(target.member).toLowerCase()] || target.member.toLowerCase().endsWith('.css')) return 'url("")';
      return `url("${resourceURL(target.member)}${target.fragment ? `#${encodeURIComponent(target.fragment)}` : ''}")`;
    } catch { return 'url("")'; }
  });
}

const TAGS = new Set('html head body title p div span section article header footer main aside nav h1 h2 h3 h4 h5 h6 ol ul li dl dt dd blockquote pre code em strong b i u s sub sup br hr figure figcaption img picture table thead tbody tfoot tr th td caption col colgroup a abbr cite q small time ruby rt rp link style font center address details summary mark kbd samp var tt big'.split(' '));
const SVG_TAGS = new Set('svg g path rect circle ellipse line polyline polygon text tspan defs clipPath linearGradient radialGradient stop use image title desc'.split(' ').map(t => t.toLowerCase()));
const ATTRS = new Set('id xml:id class lang xml:lang dir title alt width height colspan rowspan scope start type hidden role aria-label aria-hidden epub:type'.split(' '));
const SVG_ATTRS = new Set('viewbox xmlns d x y x1 y1 x2 y2 cx cy r rx ry points fill stroke stroke-width opacity fill-opacity stroke-opacity transform offset stop-color stop-opacity font-size font-family text-anchor preserveaspectratio'.split(' '));

export async function sanitizeOriginalMarkup(bytes, {member, resourceURL, sectionURL, svgOnly = false}) {
  check(bytes.length <= LIMITS.markup, 'Original document exceeds rendering limit');
  const text = decodeMarkup(bytes);
  check(!/<!ENTITY\b|<!DOCTYPE[^>]*\[/i.test(text), 'Unsafe entity declaration in original');
  const {Parser} = await import('htmlparser2');
  const root = {tag:'', attrs:{}, children:[]}; const stack=[root]; let count=0;
  const parser = new Parser({onopentag(tag,attrs) {
    check(++count <= 100000 && stack.length < 128, 'Original document structure exceeds rendering limit');
    const node = {tag:tag.toLowerCase().split(':').at(-1),attrs,children:[]}; stack.at(-1).children.push(node); stack.push(node);
  }, onclosetag(){if(stack.length>1)stack.pop();}, ontext(value){stack.at(-1).children.push(value);}},
  {decodeEntities:true, lowerCaseTags:true, lowerCaseAttributeNames:true, recognizeSelfClosing:true});
  parser.end(text);
  const serialize = (node, inSVG = false) => {
    if(typeof node==='string')return escape(node);
    if (!node.tag) return node.children.map(c => serialize(c,inSVG)).join('');
    const tag=node.tag; inSVG ||= tag==='svg';
    if (!(inSVG ? SVG_TAGS.has(tag) : TAGS.has(tag)) || (svgOnly && !inSVG)) return '';
    if(tag==='style') return `<style>${sanitizeOriginalCSS(node.children.filter(c=>typeof c==='string').join(''),member,resourceURL)}</style>`;
    const attrs=[];
    for(const [key,value] of Object.entries(node.attrs)) {
      if(ATTRS.has(key) || (tag==='a'&&key==='name') || (inSVG && SVG_ATTRS.has(key))) attrs.push(`${key === 'viewbox' ? 'viewBox' : key}="${escape(value)}"`);
      else if(key==='style')attrs.push(`style="${escape(sanitizeOriginalCSS(value,member,resourceURL))}"`);
    }
    if(!node.attrs.id&&node.attrs['xml:id'])attrs.push(`id="${escape(node.attrs['xml:id'])}"`);
    if(tag==='svg'&&!node.attrs.xmlns)attrs.push('xmlns="http://www.w3.org/2000/svg"');
    if(tag==='link') {
      if(node.attrs.rel?.toLowerCase()!=='stylesheet')return '';
      try {const target=resolveOriginalReference(member,node.attrs.href); if(path.posix.extname(target.member).toLowerCase()!=='.css')return '';
        attrs.push(`rel="stylesheet" href="${escape(resourceURL(target.member))}"`);
      } catch {return '';}
    }
    if(tag==='font')for(const key of ['color','face','size'])if(node.attrs[key])attrs.push(`${key}="${escape(node.attrs[key])}"`);
    if(tag==='img'||(inSVG&&tag==='image')) {
      try {const target=resolveOriginalReference(member,tag==='img'?node.attrs.src:node.attrs.href||node.attrs['xlink:href']); const mime=TYPES[path.posix.extname(target.member).toLowerCase()];
        if(!mime?.startsWith('image/'))return ''; attrs.push(`${tag==='img'?'src':'href'}="${escape(resourceURL(target.member))}"${tag==='img'?' loading="lazy"':''}`);
      } catch{return '';}
    }
    if(tag==='a') {
      try {const target=resolveOriginalReference(member,node.attrs.href); const url=sectionURL(target.member);
        if(url)attrs.push(`href="${escape(url)}${target.fragment ? `#${escape(encodeURIComponent(target.fragment))}`:''}" data-reader-fragment="${escape(target.fragment)}"`);
      } catch { /* External links remain text. */ }
    }
    if(inSVG && tag==='use' && /^#[A-Za-z0-9_.:-]+$/.test(node.attrs.href || node.attrs['xlink:href'] || ''))attrs.push(`href="${escape(node.attrs.href || node.attrs['xlink:href'])}"`);
    const name = ({clippath:'clipPath',lineargradient:'linearGradient',radialgradient:'radialGradient'}[tag] || tag);
    return `<${name}${attrs.length?' '+attrs.join(' '):''}>${node.children.map(c=>serialize(c,inSVG)).join('')}${['img','br','hr','col','link'].includes(tag)?'':`</${name}>`}`;
  };
  return serialize(root);
}

export function originalPdfRange(range, size) {
  let start=0;let end=size-1;
  if(range) {
    const match=/^bytes=(\d*)-(\d*)$/.exec(range);check(match&&(match[1]||match[2]),'Invalid PDF byte range',416);
    if(!match[1]) {const suffix=Number(match[2]);check(Number.isSafeInteger(suffix)&&suffix>0,'Invalid PDF byte range',416);start=Math.max(0,size-suffix);}
    else {start=Number(match[1]);if(match[2])end=Math.min(Number(match[2]),end);}
    check(Number.isSafeInteger(start)&&Number.isSafeInteger(end)&&start>=0&&start<=end&&start<size,'PDF byte range unavailable',416);
  }
  return {start,end,status:range?206:200};
}

export function createOriginalReader({store, dataDir}) {
  const assetsRoot = path.dirname(require.resolve('pdfjs-dist/package.json'));
  async function source(sectionId) {
    const section=store.db.prepare('SELECT * FROM sections WHERE id=?').get(sectionId);
    check(section,'Section not found',404); const locator=parse(section.locator_json);
    check(!locator.derived && ['pdf','epub'].includes(locator.format),'Original view is unavailable for this source format',415);
    const file=store.db.prepare('SELECT * FROM files WHERE book_id=? AND sha256=? AND format=? AND staged=1 ORDER BY id LIMIT 1').get(section.book_id,section.book_id,locator.format);
    check(file && /^[a-f0-9]{64}$/.test(file.sha256),'Verified original source is unavailable',409);
    check(!locator.sourceSha256 || locator.sourceSha256===file.sha256,'Original source identity mismatch',409);
    const root=await realpath(dataDir); const info=await lstat(file.path);
    check(info.isFile()&&!info.isSymbolicLink()&&info.size>0&&info.size<=LIMITS.source,'Original source exceeds rendering limits',413);
    const target=await realpath(file.path); check(within(root,target),'Original source is outside this library',409);
    const handle=await open(target,constants.O_RDONLY|constants.O_NOFOLLOW);
    try {
      const opened=await handle.stat();check(opened.isFile()&&opened.ino===info.ino&&opened.dev===info.dev&&opened.size===info.size,'Original source changed during opening',409);
      const digest=createHash('sha256');for await(const chunk of verifiedReadStream(handle,0))digest.update(chunk);
      const after=await handle.stat();check(after.size===opened.size&&after.mtimeMs===opened.mtimeMs&&after.ctimeMs===opened.ctimeMs,'Original source changed during verification',409);
      check(digest.digest('hex')===file.sha256,'Original source failed its identity check',409);
      if(locator.format==='pdf')check(Number.isSafeInteger(locator.page)&&locator.page>0&&locator.page<=5000,'Invalid original PDF physical page');
      else safeOriginalMember(locator.member);
      return {section,locator,file:{...file,path:target},handle,size:opened.size};
    } catch(cause) {await handle.close();throw cause;}
  }
  function send(req,res,body,type) {
    res.setHeader('Content-Type',type);res.setHeader('Content-Length',Buffer.byteLength(body));res.writeHead(200);res.end(req.method==='HEAD'?undefined:body);
  }
  return {async handle(req,res,url) {
    if(!['GET','HEAD'].includes(req.method))return false;
    if(url.pathname.startsWith('/api/reader-assets/')) {
      const name=decodeURIComponent(url.pathname.slice('/api/reader-assets/'.length));
      check(['build/pdf.mjs','build/pdf.worker.mjs'].includes(name) || /^(?:cmaps|standard_fonts|wasm)\/[a-zA-Z0-9_.-]+\.(?:bcmap|pfb|ttf|wasm|mjs|js|bin)$/.test(name),'Reader dependency not found',404);
      const target=await realpath(path.join(assetsRoot,name));check(within(assetsRoot,target),'Unsafe reader dependency',404);
      send(req,res,await readFile(target),name.endsWith('.mjs')||name.endsWith('.js')?'text/javascript; charset=utf-8':name.endsWith('.wasm')?'application/wasm':'application/octet-stream');return true;
    }
    const match=/^\/api\/sections\/([^/]+)\/original(?:\/(pdf|document|resource))?$/.exec(url.pathname);if(!match)return false;
    const verified=await source(decodeURIComponent(match[1]));const {section,locator,file,handle,size}=verified;let streaming=false;
    try {
    const base=`/api/sections/${encodeURIComponent(section.id)}/original`;
    if(!match[2]) {
      send(req,res,JSON.stringify({sectionId:section.id,bookId:section.book_id,sourceSha256:file.sha256,format:locator.format,
        ...(locator.format==='pdf'?{physicalPage:locator.page,pdfURL:`${base}/pdf`}:{member:locator.member,documentURL:`${base}/document`,reflowable:true})}), 'application/json; charset=utf-8');return true;
    }
    if(match[2]==='pdf') {
      check(locator.format==='pdf','This original source is not PDF',415);
      const range=originalPdfRange(req.headers?.range,size);
      res.setHeader('Accept-Ranges','bytes');res.setHeader('Content-Type','application/pdf');res.setHeader('Content-Length',range.end-range.start+1);
      res.setHeader('Content-Disposition',`inline; filename*=UTF-8''${encodeURIComponent(path.basename(file.relative_path)).replace(/'/g,'%27')}`);
      if(range.status===206)res.setHeader('Content-Range',`bytes ${range.start}-${range.end}/${size}`);
      res.writeHead(range.status);if(req.method==='HEAD'){res.end();return true;}
      const stream=verifiedReadStream(handle,range.start,range.end);streaming=true;
      let closed=false;const close=()=>{if(closed)return;closed=true;stream.destroy();handle.close().catch(()=>{});};
      stream.on('error',()=>{res.destroy();close();});res.once('close',close);res.once('finish',close);stream.pipe(res);return true;
    }
    check(locator.format==='epub','This original source is not EPUB',415);
    const archive=await openOriginalArchive(verified);
    try {
      const resourceURL = member => `${base}/resource?member=${encodeURIComponent(member)}`;
      const sections=store.db.prepare('SELECT id,locator_json FROM sections WHERE book_id=? ORDER BY ordinal').all(section.book_id);
      const sectionURL = member => {const found=sections.find(row=>{const l=parse(row.locator_json);return l.format==='epub'&&!l.derived&&l.member===member;});return found?`/api/sections/${encodeURIComponent(found.id)}/original/document`:null;};
      // No archive script, network request, frame, form, plugin or navigation can
      // execute. Same-origin access lets the trusted parent own position/links.
      res.setHeader('Content-Security-Policy', "sandbox allow-same-origin; default-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self'; font-src 'self'; script-src 'none'; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'");
      if(match[2]==='document') {
        const html=await sanitizeOriginalMarkup(await archive.read(locator.member,LIMITS.markup),{member:locator.member,resourceURL,sectionURL});
        send(req,res,`<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>html{color-scheme:light}body{margin:1.25rem auto;padding:0 1rem;max-width:74ch;line-height:1.6;overflow-wrap:break-word}img,svg{max-width:100%;height:auto}pre,table{max-width:100%;overflow:auto}pre{white-space:pre-wrap}</style>${html}`,'text/html; charset=utf-8');
      } else {
        const member=safeOriginalMember(url.searchParams.get('member'));const mime=TYPES[path.posix.extname(member).toLowerCase()];check(mime,'Original resource type refused',415);
        let bytes=await archive.read(member);
        if(mime.startsWith('text/css'))bytes=sanitizeOriginalCSS(decodeMarkup(bytes),member,resourceURL);
        if(mime==='image/svg+xml')bytes=await sanitizeOriginalMarkup(bytes,{member,resourceURL,sectionURL,svgOnly:true});
        send(req,res,bytes,mime);
      }
    } finally {archive.close();}
    return true;
    } finally {if(!streaming)await handle.close();}
  }};
}
