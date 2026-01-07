#!/usr/bin/env node
/**
 * Datasphere MCP Server
 *
 * Minimal MCP server that exposes datasphere query tools.
 * Shells out to `ds` CLI command.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { execSync } from 'child_process';

const SERVER_INSTRUCTIONS = `
Datasphere MCP Server - Knowledge Graph Queries

Use datasphere_query to search your distilled knowledge graph for relevant insights.
The knowledge graph contains extracted learnings from your Claude Code sessions.

Example queries:
- "How do I configure LanceDB?"
- "What's the pattern for chunking large texts?"
- "Authentication strategies we've discussed"
`.trim();

const server = new Server(
  { name: 'datasphere', version: '0.1.0' },
  {
    capabilities: { tools: {} },
    instructions: SERVER_INSTRUCTIONS
  }
);

// Define tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: 'datasphere_query',
      description: 'Search the datasphere knowledge graph for relevant insights from past sessions',
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Search query - describe what you\'re looking for'
          },
          limit: {
            type: 'number',
            description: 'Maximum number of results (default: 5)',
            default: 5
          }
        },
        required: ['query']
      }
    },
    {
      name: 'datasphere_related',
      description: 'Find nodes similar to a specific node in the knowledge graph',
      inputSchema: {
        type: 'object',
        properties: {
          node_id: {
            type: 'string',
            description: 'UUID of the node to find related nodes for'
          },
          limit: {
            type: 'number',
            description: 'Maximum number of results (default: 5)',
            default: 5
          }
        },
        required: ['node_id']
      }
    }
  ]
}));

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === 'datasphere_query') {
    const { query, limit = 5 } = args;

    try {
      const result = execSync(
        `ds query --format json --limit ${limit} ${JSON.stringify(query)}`,
        { encoding: 'utf-8', timeout: 30000 }
      );

      const nodes = JSON.parse(result);

      if (nodes.length === 0) {
        return {
          content: [{
            type: 'text',
            text: 'No relevant results found in the knowledge graph.'
          }]
        };
      }

      const formatted = nodes.map((node, i) =>
        `## Result ${i + 1} (similarity: ${node.similarity.toFixed(2)})\n` +
        `Source: ${node.source}\n` +
        `Time: ${node.timestamp}\n\n` +
        `${node.content}`
      ).join('\n\n---\n\n');

      return {
        content: [{ type: 'text', text: formatted }]
      };

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        content: [{ type: 'text', text: `Error querying datasphere: ${message}` }],
        isError: true
      };
    }
  }

  if (name === 'datasphere_related') {
    const { node_id, limit = 5 } = args;

    try {
      const result = execSync(
        `ds related --format json --limit ${limit} ${node_id}`,
        { encoding: 'utf-8', timeout: 30000 }
      );

      const nodes = JSON.parse(result);

      if (nodes.length === 0) {
        return {
          content: [{
            type: 'text',
            text: 'No related nodes found.'
          }]
        };
      }

      const formatted = nodes.map((node, i) =>
        `## Related ${i + 1} (similarity: ${node.similarity.toFixed(2)})\n` +
        `ID: ${node.id}\n` +
        `Source: ${node.source}\n` +
        `Time: ${node.timestamp}\n\n` +
        `${node.content}`
      ).join('\n\n---\n\n');

      return {
        content: [{ type: 'text', text: formatted }]
      };

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      return {
        content: [{ type: 'text', text: `Error finding related nodes: ${message}` }],
        isError: true
      };
    }
  }

  return {
    content: [{ type: 'text', text: `Unknown tool: ${name}` }],
    isError: true
  };
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Datasphere MCP server running on stdio');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
