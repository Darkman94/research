# 4chan Post Grabber - Obsidian Plugin

An Obsidian plugin that fetches 4chan threads and saves them as formatted notes in your vault.

## Features

- Fetch any 4chan thread by board and thread ID
- Save threads as well-formatted Markdown notes
- Includes original post and all replies
- Optional YAML frontmatter with thread metadata
- Image information with direct URLs
- Configurable output folder
- Preserves post structure and formatting
- Handles HTML entities and greentext

## Installation

### Manual Installation

1. Download the latest release files:
   - `main.js`
   - `manifest.json`
   - `styles.css` (if available)

2. Create a folder in your Obsidian vault's plugins folder:
   - Navigate to `{YourVault}/.obsidian/plugins/`
   - Create a new folder called `4chan-post-grabber`
   - Place the downloaded files in this folder

3. Reload Obsidian

4. Go to Settings → Community Plugins → Enable "4chan Post Grabber"

### Development Installation

1. Clone this repository into your vault's plugins folder:
   ```bash
   cd {YourVault}/.obsidian/plugins/
   git clone https://github.com/Darkman94/research.git
   cd research/obsidian-4chan-plugin
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Build the plugin:
   ```bash
   npm run build
   ```

4. Reload Obsidian and enable the plugin in Settings

## Usage

### Grabbing a 4chan Thread

1. **Using the Ribbon Icon:**
   - Click the download icon in the left sidebar
   - Enter the board name (e.g., "g", "pol", "b")
   - Enter the thread ID (the number in the thread URL)
   - Click "Grab Post"

2. **Using Command Palette:**
   - Press `Ctrl/Cmd + P` to open the command palette
   - Type "Grab 4chan post" and select it
   - Follow the same steps as above

### Finding Board and Thread ID

Given a 4chan URL like: `https://boards.4chan.org/g/thread/98765432`
- **Board**: `g`
- **Thread ID**: `98765432`

### Example

For the URL: `https://boards.4chan.org/pol/thread/123456789`

1. Open the 4chan Post Grabber modal
2. Enter board: `pol`
3. Enter thread ID: `123456789`
4. Click "Grab Post"

The plugin will:
- Fetch the thread from 4chan's API
- Create a formatted note in your configured folder
- Open the note automatically

## Settings

Access settings via: Settings → 4chan Post Grabber

### Available Settings

- **Output folder**: Where notes will be saved (default: `4chan`)
- **Include image information**: Toggle image filenames and URLs in notes
- **Include metadata**: Toggle YAML frontmatter with thread metadata

## Note Format

Generated notes include:

### YAML Frontmatter (if enabled)
```yaml
---
board: g
thread: 98765432
url: https://boards.4chan.org/g/thread/98765432
fetched: 2025-11-08T12:00:00.000Z
posts: 150
unique_ips: 45
---
```

### Thread Header
- Thread title (if available)
- Board and thread ID
- Posted timestamp
- Poster information (name, tripcode, ID)

### Original Post
- Full post content
- Image information (if applicable)

### Replies
Each reply includes:
- Post number
- Timestamp and poster info
- Post content
- Image information (if applicable)

## API Usage and Rate Limits

This plugin uses the official 4chan API (https://github.com/4chan/4chan-API).

**Important**: The 4chan API has rate limits:
- Maximum 1 request per second
- Be respectful of the API

The plugin automatically respects these limits for single requests. Avoid rapid consecutive fetches.

## Privacy and Security

- All requests go directly to 4chan's official API (a.4cdn.org)
- No third-party services or tracking
- No data is stored except in your local Obsidian vault
- Images are not downloaded, only URLs are saved

## Troubleshooting

### "HTTP error! status: 404"
- The thread may have been deleted or archived
- Double-check the board name and thread ID

### "No posts found in thread!"
- The API returned an empty response
- The thread may have been removed

### Plugin doesn't appear in settings
- Make sure files are in the correct folder
- Reload Obsidian (Ctrl/Cmd + R)
- Check the console for errors (Ctrl/Cmd + Shift + I)

## Development

### Building from Source

```bash
# Install dependencies
npm install

# Development build with watch mode
npm run dev

# Production build
npm run build
```

### Project Structure

- `main.ts` - Main plugin code
- `manifest.json` - Plugin metadata
- `package.json` - Dependencies and scripts
- `tsconfig.json` - TypeScript configuration
- `esbuild.config.mjs` - Build configuration

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Disclaimer

This plugin is not affiliated with or endorsed by 4chan. Use responsibly and follow 4chan's terms of service and API guidelines.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions

## Changelog

### Version 1.0.0
- Initial release
- Fetch threads by board and ID
- Save as formatted Markdown notes
- Configurable settings
- Image URL support
- YAML frontmatter metadata
