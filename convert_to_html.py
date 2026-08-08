import markdown

with open('Submission_Report.md', 'r') as f:
    text = f.read()

html_content = markdown.markdown(text, extensions=['fenced_code'])

full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            line-height: 1.6;
            color: #333;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

with open('Submission_Report.html', 'w') as f:
    f.write(full_html)
