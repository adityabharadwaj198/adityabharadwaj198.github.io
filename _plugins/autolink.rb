# Kramdown doesn't linkify bare URLs; wrap them in <...> before rendering.
# Skips fenced code blocks, inline code, and URLs already part of markdown/HTML links.
BARE_URL = %r{(?<![(<\["'=\w/])(https?://[^\s<>)\]"']+[^\s<>)\]"'.,;:!?])}

Jekyll::Hooks.register [:posts, :documents], :pre_render do |doc|
  next unless doc.extname == ".md"

  in_fence = false
  doc.content = doc.content.lines.map do |line|
    if line =~ /^\s*(```|~~~)/
      in_fence = !in_fence
      next line
    end
    next line if in_fence || line =~ /^( {4}|\t)/

    line.split(/(`[^`]*`)/).map do |part|
      part.start_with?("`") ? part : part.gsub(BARE_URL, '<\1>')
    end.join
  end.join
end
