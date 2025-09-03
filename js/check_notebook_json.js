import fs from 'fs';
import { lint } from 'markdownlint/sync';

const expectedMetadataKeys = new Set(['kernelspec', 'language_info']);

function checkOutput(nbPath, nb) {
  const cells = nb.cells || [];
  for (const cell of cells) {
    if (cell.cell_type === 'code' && cell.outputs && cell.outputs.length > 0) {
      throw new Error(`Notebook ${nbPath} has output!`);
    }
  }
}

function checkEmptyCells(nbPath, nb) {
  const cells = nb.cells || [];
  for (const cell of cells) {
    const source = cell.source || [];
    if (source.length === 0 || (source.length === 1 && source[0] === '')) {
      throw new Error(`Notebook ${nbPath} has empty cells!`);
    }
  }
}

function checkMetadataKeys(nbPath, nb) {
  if (!nb.metadata) {
    throw new Error('Notebook metadata is missing');
  }

  const metadataKeys = new Set(Object.keys(nb.metadata));
  const expectedKeys = Array.from(expectedMetadataKeys).sort();
  const actualKeys = Array.from(metadataKeys).sort();

  if (expectedKeys.join(',') !== actualKeys.join(',')) {
    throw new Error(
      `Metadata keys in ${nbPath} are incorrect: ` +
      `{${actualKeys.join(', ')}} != {${expectedKeys.join(', ')}}`
    );
  }
}

function checkMarkdownCells(nbPath, nb) {
  const cells = nb.cells || [];
  const markdownCells = cells.filter(cell => cell.cell_type === 'markdown');

  if (markdownCells.length === 0) {
    return; // No markdown cells to check
  }

  const errors = [];

  markdownCells.forEach((cell, index) => {
    const source = cell.source || [];
    if (source.length === 0) {
      return; // Skip empty cells (handled by checkEmptyCells)
    }

    // Join source lines into a single string
    const markdownContent = Array.isArray(source) ? source.join('') : source;

    // Create a temporary object for markdownlint
    const options = {
      strings: {
        [`cell_${index}`]: markdownContent
      },
      config: {
        'default': true,
        'MD047': false, // Don't require newline at end of "file"
        'MD013': { 'line_length': 91 },
        'MD041': false // Don't require H1 as first line
      }
    };

    const result = lint(options);

    if (result[`cell_${index}`] && result[`cell_${index}`].length > 0) {
      result[`cell_${index}`].forEach(error => {
        // Get the specific line that has the error
        const lines = markdownContent.split('\n');
        const errorLine = lines[error.lineNumber - 1] || '';

        errors.push(
          `Markdown cell ${index + 1}: ${error.ruleDescription} ` +
            `(${error.ruleNames.join('/')}) at line ${error.lineNumber}\n` +
            `  Text: "${errorLine}"`
        );
      });
    }
  });

  if (errors.length > 0) {
    throw new Error(`Markdown linting errors in ${nbPath}:\n${errors.join('\n')}`);
  }
}

function main() {
  if (process.argv.length !== 3) {
    console.error('Usage: node check_notebook_json.js <notebook_path>');
    process.exit(1);
  }

  const notebookPath = process.argv[2];

  try {
    const fileContent = fs.readFileSync(notebookPath, 'utf8');
    // Collect all lines exceeding 100 characters.
    const lines = fileContent.split('\n');
    const longLines = [];
    for (const [index, line] of lines.entries()) {
      if (line.length > 100) {
        longLines.push(`Line ${index + 1}: ${line}`);
      }
    }
    if (longLines.length > 0) {
      throw new Error(
        `The following lines in ${notebookPath} exceed 100 characters:\n` +
      longLines.join('\n')
      );
    }
    const nb = JSON.parse(fileContent);

    checkOutput(notebookPath, nb);
    checkEmptyCells(notebookPath, nb);
    checkMetadataKeys(notebookPath, nb);
    checkMarkdownCells(notebookPath, nb);

    console.log(`Notebook ${notebookPath} validation passed!`);
  } catch (error) {
    console.error(`Error: ${error.message}`);
    process.exit(1);
  }
}

main();
