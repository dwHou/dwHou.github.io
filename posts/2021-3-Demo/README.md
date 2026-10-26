# Adding a demo

1. Copy `demo-template.html` to the next `demoN.html` (for example, `demo11.html`).
2. Edit the marked title, description, introduction, and video path. Put the MP4 in `video/` and name it `demoN_short_name.mp4`.
3. Add one `<li><a href="posts/2021-3-Demo/demoN.html" data-year="2026">Demo title</a></li>` to the `demo-card-list` in the root `demo.html`. Change `data-year` to the approximate year the demo was created.

The template uses the blog's existing navigation and styles. For a two-video comparison, copy the video section from `demo9.html`. The index cards are styled automatically; no thumbnail or build script is needed.
