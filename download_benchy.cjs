const fs = require('fs');
async function run() {
  const res = await fetch("https://api.github.com/search/code?q=filename:3DBenchy.stl", { headers: { "User-Agent": "NodeJS" } });
  const data = await res.json();
  if (data.items && data.items.length > 0) {
    for (const item of data.items) {
      const rawUrl = item.html_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/");
      console.log("Downloading from", rawUrl);
      const stlRes = await fetch(rawUrl);
      if (stlRes.status === 200) {
        const buffer = await stlRes.arrayBuffer();
        if (buffer.byteLength > 1000) {
          fs.writeFileSync("public/models/3DBenchy.stl", Buffer.from(buffer));
          console.log("Saved. Size:", buffer.byteLength);
          return;
        }
      }
    }
  } else {
    console.log("No items found");
  }
}
run();
