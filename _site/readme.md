To make changes to the styling of your website, you should now edit the file:
_sass/_main.scss
Here's why:
I moved all the actual CSS rules into _sass/_main.scss. This file contains all the definitions for fonts, colors, layouts, and other visual aspects of your site.
The assets/css/style.scss file now acts as an "entry point" for Jekyll. It only contains the necessary Jekyll front matter (---) and an @import "main"; directive. This tells Jekyll to look in the _sass directory for _main.scss and compile it into the final style.css file that your browser reads.
So, if you want to change colors, fonts, spacing, or any other visual elements, make your edits in _sass/_main.scss.
Important: After making any changes to _sass/_main.scss, remember to:
Stop your Jekyll server (if it's running) by pressing Ctrl + C in the terminal.
Restart the Jekyll server by running bundle exec jekyll serve.
Hard refresh your browser (Ctrl + Shift + R or Cmd + Shift + R) to see the changes reflected.
This separation helps keep your project organized and follows Jekyll's best practices for SASS management.

You're asking excellent questions! It's important to know how to manage your content.

Here's how you can do both:

### How to Change the Contents of Your Website

Your main website content (like your "About Me" section, "Connect With Me," and "Tech Stack") is managed in Markdown files, primarily `index.md`.

*   **For your main page content**:
    *   Edit the `index.md` file located in the root of your project.
    *   You can change any text, add new sections, update links, etc., by directly editing the Markdown.

*   **For other static pages (like an "About" page)**:
    *   If you decide to create a separate "About" page (e.g., `about.md`), you would edit that file directly.

**Remember:** After making any changes to these Markdown files, you'll need to:
1.  **Stop your Jekyll server** (if it's running) by pressing `Ctrl + C` in the terminal.
2.  **Restart the Jekyll server** by running `bundle exec jekyll serve`.
3.  **Hard refresh your browser** (`Ctrl + Shift + R` or `Cmd + Shift + R`) to see the updates.

### How to Add New Blog Posts

Adding new blog posts is a straightforward process in Jekyll:

1.  **Create a new file in the `_posts` directory**:
    *   Navigate to the `_posts` directory in your project.
    *   Create a new Markdown file (`.md` or `.markdown`) with a specific naming convention: `YYYY-MM-DD-your-post-title.md`.
    *   **Example:** For a post about "My First Project" published today (June 13, 2025), you would create `_posts/2025-06-13-my-first-project.md`.

2.  **Add "Front Matter" to the top of your post**:
    *   Every blog post needs a block of YAML code at the very top, enclosed by `---` lines. This is called "Front Matter" and provides metadata about your post.
    *   Here's a basic example of what you should put at the top of your new blog post file:

    ```yaml
    ---
    layout: post
    title: "Your Awesome Blog Post Title"
    date: YYYY-MM-DD # Make sure this matches the date in your filename
    categories: [technology, programming] # Add relevant categories
    tags: [jekyll, webdev, thoughts] # Add relevant tags
    excerpt: "This is a short summary of your blog post. It will appear on the blog listing page."
    ---
    ```
    *   **`layout: post`**: This tells Jekyll to use the `_layouts/post.html` template we created for your blog posts.
    *   **`title`**: The title of your blog post.
    *   **`date`**: The publication date of your post.
    *   **`categories`**: A list of categories for your post (e.g., "Technology", "Personal"). These will appear on the blog listing.
    *   **`tags`**: A list of keywords or tags for your post (e.g., "Jekyll", "Tutorial").
    *   **`excerpt`**: A short summary that will be displayed on the main blog listing page (`/blog/`). If omitted, Jekyll will automatically generate one from the beginning of your post.

3.  **Write your content below the Front Matter**:
    *   After the second `---` line, you can write your blog post content using Markdown.

**After creating a new blog post file:**
1.  **Stop your Jekyll server** (if it's running) by pressing `Ctrl + C` in the terminal.
2.  **Restart the Jekyll server** by running `bundle exec jekyll serve`.
3.  Your new post will automatically appear on the `/blog/` page and will be accessible at its unique URL (e.g., `/blog/2025/06/13/my-first-project/`).

Let me know if you have any more questions about content creation!