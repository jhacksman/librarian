// Original, deliberately small EPUBs for the Spark recovery workflow only.
import {crc32} from 'node:zlib';

export const BOOKS = Object.freeze([
  {filename: '01-lantern.epub', title: 'The Amber Lantern', sections: [
    {title: 'Lantern timing', paragraph: 'The amber lantern burns for exactly 42 minutes. The keeper records the interval in the copper ledger.'},
    {title: 'Lantern storage', paragraph: 'Store the amber lantern beside the eastern window. A blue cloth protects its polished handle.'},
  ]},
  {filename: '02-compass.epub', title: 'The Silver Compass', sections: [
    {title: 'Compass calibration', paragraph: 'The silver compass is calibrated at dawn. Its keeper turns the dial exactly seven times.'},
    {title: 'Compass storage', paragraph: 'The silver compass rests in a cedar box. A green ribbon identifies the north-facing edge.'},
  ]},
]);

export const expectedSection = (section) => `${section.title}\n\n${section.paragraph}`;

export function syntheticEpub(book) {
  const entries = [
    ['mimetype', 'application/epub+zip'],
    ['META-INF/container.xml', '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OPS/book.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'],
    ['OPS/book.opf', `<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>${book.title}</dc:title><dc:creator>Recovery Fixture</dc:creator><dc:identifier id="id">urn:recovery:${book.filename}</dc:identifier><dc:language>en</dc:language></metadata><manifest>${book.sections.map((_, i) => `<item id="s${i + 1}" href="s${i + 1}.xhtml" media-type="application/xhtml+xml"/>`).join('')}</manifest><spine>${book.sections.map((_, i) => `<itemref idref="s${i + 1}"/>`).join('')}</spine></package>`],
    ...book.sections.map((section, i) => [`OPS/s${i + 1}.xhtml`, `<html xmlns="http://www.w3.org/1999/xhtml"><head><title>${section.title}</title></head><body><h1 id="heading">${section.title}</h1><p>${section.paragraph}</p></body></html>`]),
  ];
  const locals = []; const centrals = []; let offset = 0;
  for (const [filename, text] of entries) {
    const name = Buffer.from(filename); const bytes = Buffer.from(text); const checksum = crc32(bytes);
    const local = Buffer.alloc(30); const central = Buffer.alloc(46);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x800, 6);
    local.writeUInt32LE(checksum, 14); local.writeUInt32LE(bytes.length, 18); local.writeUInt32LE(bytes.length, 22);
    local.writeUInt16LE(name.length, 26);
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE((3 << 8) | 20, 4);
    central.writeUInt16LE(20, 6); central.writeUInt16LE(0x800, 8); central.writeUInt32LE(checksum, 16);
    central.writeUInt32LE(bytes.length, 20); central.writeUInt32LE(bytes.length, 24);
    central.writeUInt16LE(name.length, 28); central.writeUInt32LE((0o100644 << 16) >>> 0, 38);
    central.writeUInt32LE(offset, 42);
    locals.push(local, name, bytes); centrals.push(central, name); offset += local.length + name.length + bytes.length;
  }
  const directory = Buffer.concat(centrals); const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, directory, end]);
}
