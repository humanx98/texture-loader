# Download stb_image.h

Standalone builds now restore pinned stb headers automatically through vcpkg.
These bundled headers and the manual download instructions below are only
needed when configuring with `-DUSE_VCPKG=OFF`. See [BUILD.md](../../BUILD.md).

This directory should contain stb_image.h from the stb library.

## Download:

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -OutFile "stb_image.h"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" -OutFile "external\stb\stb_image_write.h"
```

**Linux/macOS:**
```bash
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
wget https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
# or
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -O https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

```

## Alternative:

Visit https://github.com/nothings/stb and download stb_image.h manually.
