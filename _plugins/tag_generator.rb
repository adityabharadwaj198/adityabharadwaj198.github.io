module Jekyll
  class TagPageGenerator < Generator
    safe true
    priority :normal

    def generate(site)
      tags = site.posts.docs.flat_map { |post| post.data['tags'] || [] }.uniq
      tags.each do |tag|
        site.pages << TagPage.new(site, site.source, 'tags', tag)
      end
    end
  end

  class TagPage < Page
    def initialize(site, base, dir, tag)
      @site = site
      @base = base
      @dir = dir
      @name = "#{tag.downcase.gsub(/[^a-z0-9]+/, '-')}/index.html"

      self.process(@name)
      self.read_yaml(File.join(base, '_layouts'), 'tag.html')
      self.data['tag'] = tag
      self.data['title'] = "Posts tagged with \"#{tag}\""
      self.data['posts'] = site.posts.docs.select { |post| post.data['tags']&.include?(tag) }
    end
  end
end 