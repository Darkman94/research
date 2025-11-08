import { App, Modal, Notice, Plugin, PluginSettingTab, Setting, TFile } from 'obsidian';

// Interface for plugin settings
interface FourChanPluginSettings {
	outputFolder: string;
	includeImages: boolean;
	includeMetadata: boolean;
}

// Default settings
const DEFAULT_SETTINGS: FourChanPluginSettings = {
	outputFolder: '4chan',
	includeImages: true,
	includeMetadata: true
}

// Interface for 4chan post structure
interface FourChanPost {
	no: number;
	resto: number;
	now: string;
	time: number;
	name: string;
	com?: string;
	sub?: string;
	trip?: string;
	id?: string;
	capcode?: string;
	country?: string;
	country_name?: string;
	filename?: string;
	ext?: string;
	w?: number;
	h?: number;
	tn_w?: number;
	tn_h?: number;
	tim?: number;
	md5?: string;
	fsize?: number;
	replies?: number;
	images?: number;
	unique_ips?: number;
}

// Interface for 4chan thread response
interface FourChanThread {
	posts: FourChanPost[];
}

export default class FourChanPlugin extends Plugin {
	settings: FourChanPluginSettings;

	async onload() {
		await this.loadSettings();

		// Add ribbon icon
		this.addRibbonIcon('download', 'Grab 4chan post', () => {
			new FourChanModal(this.app, this).open();
		});

		// Add command to grab 4chan post
		this.addCommand({
			id: 'grab-4chan-post',
			name: 'Grab 4chan post',
			callback: () => {
				new FourChanModal(this.app, this).open();
			}
		});

		// Add settings tab
		this.addSettingTab(new FourChanSettingTab(this.app, this));
	}

	onunload() {
		// Cleanup if needed
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
	}

	// Fetch thread from 4chan API
	async fetchThread(board: string, threadId: string): Promise<FourChanThread> {
		const url = `https://a.4cdn.org/${board}/thread/${threadId}.json`;

		try {
			const response = await fetch(url);

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			const data = await response.json();
			return data as FourChanThread;
		} catch (error) {
			console.error('Error fetching 4chan thread:', error);
			throw error;
		}
	}

	// Convert HTML entities in text
	decodeHtml(html: string): string {
		const txt = document.createElement('textarea');
		txt.innerHTML = html;
		return txt.value;
	}

	// Strip HTML tags but preserve structure
	stripHtml(html: string): string {
		// Convert <br> tags to newlines
		let text = html.replace(/<br\s*\/?>/gi, '\n');
		// Convert <p> tags to newlines
		text = text.replace(/<p[^>]*>/gi, '\n');
		text = text.replace(/<\/p>/gi, '\n');
		// Handle quotes (greentext)
		text = text.replace(/<span class="quote">(.*?)<\/span>/gi, '$1');
		// Remove all other HTML tags
		text = text.replace(/<[^>]+>/g, '');
		// Decode HTML entities
		text = this.decodeHtml(text);
		// Clean up multiple newlines
		text = text.replace(/\n{3,}/g, '\n\n');
		return text.trim();
	}

	// Create note from thread
	async createNoteFromThread(board: string, threadId: string) {
		try {
			new Notice('Fetching 4chan thread...');

			const thread = await this.fetchThread(board, threadId);

			if (!thread.posts || thread.posts.length === 0) {
				new Notice('No posts found in thread!');
				return;
			}

			const op = thread.posts[0];
			const replies = thread.posts.slice(1);

			// Build note content
			let content = '';

			// Add metadata if enabled
			if (this.settings.includeMetadata) {
				content += '---\n';
				content += `board: ${board}\n`;
				content += `thread: ${threadId}\n`;
				content += `url: https://boards.4chan.org/${board}/thread/${threadId}\n`;
				content += `fetched: ${new Date().toISOString()}\n`;
				content += `posts: ${thread.posts.length}\n`;
				if (op.unique_ips) content += `unique_ips: ${op.unique_ips}\n`;
				content += '---\n\n';
			}

			// Add title
			const title = op.sub ? this.stripHtml(op.sub) : `Thread ${threadId}`;
			content += `# ${title}\n\n`;

			// Add OP info
			content += `**Board:** /${board}/\n`;
			content += `**Thread:** ${threadId}\n`;
			content += `**Posted:** ${op.now}\n`;
			if (op.name) content += `**Name:** ${op.name}\n`;
			if (op.trip) content += `**Trip:** ${op.trip}\n`;
			if (op.id) content += `**ID:** ${op.id}\n`;
			content += '\n---\n\n';

			// Add OP post
			if (op.com) {
				content += '## Original Post\n\n';
				content += this.stripHtml(op.com) + '\n\n';
			}

			// Add image info if present and enabled
			if (this.settings.includeImages && op.filename && op.ext) {
				content += `**File:** ${op.filename}${op.ext}`;
				if (op.w && op.h) content += ` (${op.w}x${op.h})`;
				if (op.fsize) content += ` - ${(op.fsize / 1024).toFixed(2)} KB`;
				content += '\n';
				const imageUrl = `https://i.4cdn.org/${board}/${op.tim}${op.ext}`;
				content += `**Image URL:** ${imageUrl}\n\n`;
			}

			// Add replies
			if (replies.length > 0) {
				content += '---\n\n## Replies\n\n';

				for (const reply of replies) {
					content += `### Post ${reply.no}\n\n`;
					content += `**Posted:** ${reply.now}`;
					if (reply.name) content += ` | **Name:** ${reply.name}`;
					if (reply.trip) content += ` | **Trip:** ${reply.trip}`;
					if (reply.id) content += ` | **ID:** ${reply.id}`;
					content += '\n\n';

					if (reply.com) {
						content += this.stripHtml(reply.com) + '\n\n';
					}

					// Add image info if present
					if (this.settings.includeImages && reply.filename && reply.ext) {
						content += `**File:** ${reply.filename}${reply.ext}`;
						if (reply.w && reply.h) content += ` (${reply.w}x${reply.h})`;
						content += '\n';
						const imageUrl = `https://i.4cdn.org/${board}/${reply.tim}${reply.ext}`;
						content += `**Image URL:** ${imageUrl}\n\n`;
					}

					content += '---\n\n';
				}
			}

			// Create folder if it doesn't exist
			const folder = this.settings.outputFolder;
			if (folder && !await this.app.vault.adapter.exists(folder)) {
				await this.app.vault.createFolder(folder);
			}

			// Create filename
			const sanitizedTitle = title.replace(/[\\/:*?"<>|]/g, '-').substring(0, 100);
			const filename = `${folder}/${board}-${threadId}-${sanitizedTitle}.md`;

			// Check if file already exists
			const existingFile = this.app.vault.getAbstractFileByPath(filename);
			if (existingFile instanceof TFile) {
				// Update existing file
				await this.app.vault.modify(existingFile, content);
				new Notice('Note updated successfully!');
			} else {
				// Create new file
				await this.app.vault.create(filename, content);
				new Notice('Note created successfully!');
			}

			// Open the file
			const file = this.app.vault.getAbstractFileByPath(filename);
			if (file instanceof TFile) {
				await this.app.workspace.getLeaf(false).openFile(file);
			}

		} catch (error) {
			console.error('Error creating note:', error);
			new Notice(`Error: ${error.message}`);
		}
	}
}

// Modal for user input
class FourChanModal extends Modal {
	plugin: FourChanPlugin;
	board: string = '';
	threadId: string = '';

	constructor(app: App, plugin: FourChanPlugin) {
		super(app);
		this.plugin = plugin;
	}

	onOpen() {
		const { contentEl } = this;
		contentEl.empty();

		contentEl.createEl('h2', { text: 'Grab 4chan Post' });

		// Board input
		new Setting(contentEl)
			.setName('Board')
			.setDesc('Enter the board name (e.g., "g", "pol", "b")')
			.addText(text => text
				.setPlaceholder('g')
				.onChange(value => {
					this.board = value.trim();
				}));

		// Thread ID input
		new Setting(contentEl)
			.setName('Thread ID')
			.setDesc('Enter the thread ID (the number in the URL)')
			.addText(text => text
				.setPlaceholder('123456789')
				.onChange(value => {
					this.threadId = value.trim();
				}));

		// Submit button
		new Setting(contentEl)
			.addButton(btn => btn
				.setButtonText('Grab Post')
				.setCta()
				.onClick(() => {
					if (!this.board || !this.threadId) {
						new Notice('Please enter both board and thread ID');
						return;
					}
					this.close();
					this.plugin.createNoteFromThread(this.board, this.threadId);
				}));
	}

	onClose() {
		const { contentEl } = this;
		contentEl.empty();
	}
}

// Settings tab
class FourChanSettingTab extends PluginSettingTab {
	plugin: FourChanPlugin;

	constructor(app: App, plugin: FourChanPlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();

		containerEl.createEl('h2', { text: '4chan Post Grabber Settings' });

		// Output folder setting
		new Setting(containerEl)
			.setName('Output folder')
			.setDesc('Folder where 4chan notes will be saved')
			.addText(text => text
				.setPlaceholder('4chan')
				.setValue(this.plugin.settings.outputFolder)
				.onChange(async (value) => {
					this.plugin.settings.outputFolder = value;
					await this.plugin.saveSettings();
				}));

		// Include images setting
		new Setting(containerEl)
			.setName('Include image information')
			.setDesc('Include image filenames and URLs in notes')
			.addToggle(toggle => toggle
				.setValue(this.plugin.settings.includeImages)
				.onChange(async (value) => {
					this.plugin.settings.includeImages = value;
					await this.plugin.saveSettings();
				}));

		// Include metadata setting
		new Setting(containerEl)
			.setName('Include metadata')
			.setDesc('Include YAML frontmatter with thread metadata')
			.addToggle(toggle => toggle
				.setValue(this.plugin.settings.includeMetadata)
				.onChange(async (value) => {
					this.plugin.settings.includeMetadata = value;
					await this.plugin.saveSettings();
				}));
	}
}
