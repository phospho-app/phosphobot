// src/app/api/hf-proxy/[...path]/route.ts
import { NextRequest, NextResponse } from 'next/server';

const HF_TOKEN = process.env.HUGGINGFACE_TOKEN;
const HF_BASE_URL = 'https://huggingface.co';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  if (!HF_TOKEN) {
    return NextResponse.json(
      { error: 'HuggingFace token not configured' },
      { status: 500 }
    );
  }

  const path = params.path.join('/');
  const url = `${HF_BASE_URL}/${path}`;
  
  try {
    const response = await fetch(url, {
      headers: {
        'Authorization': `Bearer ${HF_TOKEN}`,
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Failed to fetch: ${response.statusText}` },
        { status: response.status }
      );
    }

    const contentType = response.headers.get('content-type');
    
    // Handle different content types
    if (contentType?.includes('application/json')) {
      const data = await response.json();
      return NextResponse.json(data);
    } else if (contentType?.includes('video')) {
      // For video files, return the binary data
      const buffer = await response.arrayBuffer();
      return new NextResponse(buffer, {
        headers: {
          'Content-Type': contentType,
        },
      });
    } else {
      // For other binary files (like parquet)
      const buffer = await response.arrayBuffer();
      return new NextResponse(buffer, {
        headers: {
          'Content-Type': contentType || 'application/octet-stream',
        },
      });
    }
  } catch (error) {
    console.error('Proxy error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from HuggingFace' },
      { status: 500 }
    );
  }
}