from bs4 import BeautifulSoup

html = '''
<html>
    <body>
        <h1>Title</h1>
        <p>This is a paragraph.</p>
        <a href="https://example.com">Link</a>
    </body>
</html>
'''

soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text(separator=' ')

print(text)