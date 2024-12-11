from pathlib import Path

root_path = Path(__file__).parent.parent

def read_html(relative_path_str: str) -> str:
    """
    Read the contents of an HTML file and return it.
    
    Args:
        relative_path_str (str): The path of the HTML file
            to read. Note that it is a relative path, so it
            should be relative to the submodule directory
            that the file that calls this is located in.
    
    Returns:
        A string containing the contents of the HTML
            file.
    """
    return relative_file_read(relative_path_str)

def read_css(relative_path_str: str) -> str:
    """
    Read the contents of a CSS file and return it in
    HTML format.

    Args:
        relative_path_str (str): The path of a CSS file
            to read. Note that it is a relative path, so it
            should be realative to the submodule directory
            that the file that calls this is located in.
    
    Returns:
        A string containing an HTML representation of this
            CSS file using the `<style>` tag.
    """

    return '<style>' + relative_file_read(relative_path_str) + '</style>'

def relative_file_read(relative_path_str: str) -> str:
    """
    Read the contents of a provided file and return it,
    based on a relative path instead of an absolute path.
    A relative path means it is relevant to your current
    submodule directory as opposed to an absolute path that
    requires every parent directory to be listed.

    If the file `llmvis/example/example.py` wants to read
    `llmvis/example/example.txt`, it can just use
    `relative_file_read('example.txt')'` instead of having
    to provide an absolute path. While regular file I/O in
    Python does support relative paths, the relative path
    will be different if `example.py` is imported.
    `relative_file_read` returns the relative path based on
    the submodule directory's location.

    Args:
        relative_path_str (str): The relative path of the file
            that should be accessed.
    
    Returns:
        A string containing the contents of the file.
    """

    file = absolute_path(relative_path_str).open()
    return file.read()

def absolute_path(relative_path_str: str) -> Path:
    """
    Given a relative path, return the absolute path.

    Args:
        relative_path_str (str): The relative path that
            should be converted to an absolute path.
    
    Returns:
        A `Path` representing the absolute path.
    """
    return root_path / relative_path_str