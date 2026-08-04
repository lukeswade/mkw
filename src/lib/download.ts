/**
 * Triggers a browser download for a Blob.
 *
 * Lives apart from the exporters because App.tsx needs it eagerly, and the
 * exporters import three.js — pulling this from there would drag the whole
 * engine into the initial bundle.
 */
export function downloadFile(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
