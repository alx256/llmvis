from setuptools import setup, find_packages

setup(
    name = 'llmvis',
    version = '0.1',
    author = 'Alex Poulson',
    author_email = 'psya14@nottingham.ac.uk',
    description = 'Library for XAI visualizations of LLM applications',
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    packages = ['llmvis'] + ['llmvis.' + pkg for pkg in find_packages('llmvis')],
    package_data = {
        'llmvis' : ['assets/fonts/*'],
        'llmvis.visualization': ['css/*.css', 'html/*.html', 'js/*.js']
    },
    install_requires = [
        'ollama',
        'numpy',
        'scikit-learn',
        'IPython',
        'PyQt6',
        'PyQt6-WebEngine'
    ]
)
