/**
 * LDP (Linked Data Platform) header utilities
 */

import { getAcceptHeaders } from '../rdf/conneg.js';

const LDP = 'http://www.w3.org/ns/ldp#';

/**
 * Get Link headers for a resource
 * @param {boolean} isContainer
 * @param {string} aclUrl - URL to the ACL resource
 * @returns {string}
 */
export function getLinkHeader(isContainer, aclUrl = null) {
  const links = [`<${LDP}Resource>; rel="type"`];

  if (isContainer) {
    links.push(`<${LDP}Container>; rel="type"`);
    links.push(`<${LDP}BasicContainer>; rel="type"`);
  }

  // Add acl link for auxiliary resource discovery
  if (aclUrl) {
    links.push(`<${aclUrl}>; rel="acl"`);
  }

  return links.join(', ');
}

/**
 * Get the ACL URL for a resource
 * @param {string} resourceUrl - Full URL of the resource
 * @param {boolean} isContainer - Whether the resource is a container
 * @returns {string} ACL URL
 */
export function getAclUrl(resourceUrl, isContainer) {
  if (isContainer) {
    // Container ACL: /path/.acl
    const base = resourceUrl.endsWith('/') ? resourceUrl : resourceUrl + '/';
    return base + '.acl';
  }
  // Resource ACL: /path/file.acl
  return resourceUrl + '.acl';
}

/**
 * Get standard LDP response headers
 * @param {object} options
 * @returns {object}
 */
export function getResponseHeaders({ isContainer = false, etag = null, contentType = null, resourceUrl = null, wacAllow = null, connegEnabled = false, updatesVia = null }) {
  // Calculate ACL URL if resource URL provided
  const aclUrl = resourceUrl ? getAclUrl(resourceUrl, isContainer) : null;

  const headers = {
    'Link': getLinkHeader(isContainer, aclUrl),
    'Accept-Patch': 'text/n3, application/sparql-update',
    'Accept-Ranges': isContainer ? 'none' : 'bytes',
    'Allow': 'GET, HEAD, PUT, DELETE, PATCH, OPTIONS' + (isContainer ? ', POST' : ''),
    'Vary': connegEnabled ? 'Accept, Authorization, Origin' : 'Authorization, Origin'
  };

  // Only set WAC-Allow if explicitly provided (otherwise the auth hook sets it)
  if (wacAllow) {
    headers['WAC-Allow'] = wacAllow;
  }

  // Add Accept-* headers (conneg-aware)
  const acceptHeaders = getAcceptHeaders(connegEnabled, isContainer);
  Object.assign(headers, acceptHeaders);

  // Add Updates-Via header for WebSocket notifications discovery
  if (updatesVia) {
    headers['Updates-Via'] = updatesVia;
  }

  if (etag) {
    headers['ETag'] = etag;
  }

  if (contentType) {
    headers['Content-Type'] = contentType;
  }

  return headers;
}

// CORS origin allowlist (populated from CORS_ALLOWED_ORIGINS env var)
const corsAllowedOrigins = (() => {
  const envOrigins = process.env.CORS_ALLOWED_ORIGINS;
  if (!envOrigins) return null; // null = open Solid-compatible CORS
  return new Set(envOrigins.split(',').map(o => o.trim()).filter(Boolean));
})();

/**
 * Get CORS headers
 * When CORS_ALLOWED_ORIGINS is set, only those origins are allowed.
 * When unset, Solid-compatible open CORS is used (origin reflected, not wildcard,
 * because credentials require explicit origin).
 * @param {string} origin
 * @returns {object}
 */
export function getCorsHeaders(origin) {
  const allowedMethods = 'GET, HEAD, POST, PUT, DELETE, PATCH, OPTIONS';
  const allowedHeaders = 'Accept, Authorization, Content-Type, DPoP, If-Match, If-None-Match, Link, Range, Slug, Origin';
  const exposedHeaders = 'Accept-Patch, Accept-Post, Accept-Ranges, Allow, Content-Length, Content-Range, Content-Type, ETag, Link, Location, Updates-Via, WAC-Allow';

  if (corsAllowedOrigins) {
    // Restrictive mode: only configured origins allowed
    if (origin && corsAllowedOrigins.has(origin)) {
      return {
        'Access-Control-Allow-Origin': origin,
        'Access-Control-Allow-Methods': allowedMethods,
        'Access-Control-Allow-Headers': allowedHeaders,
        'Access-Control-Expose-Headers': exposedHeaders,
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Max-Age': '86400'
      };
    }
    // Unknown origin: restrictive headers (no Allow-Origin)
    return {
      'Access-Control-Allow-Methods': allowedMethods,
      'Access-Control-Allow-Headers': allowedHeaders,
      'Access-Control-Max-Age': '86400'
    };
  }

  // Open Solid-compatible CORS (no allowlist configured)
  return {
    'Access-Control-Allow-Origin': origin || '*',
    'Access-Control-Allow-Methods': allowedMethods,
    'Access-Control-Allow-Headers': allowedHeaders,
    'Access-Control-Expose-Headers': exposedHeaders,
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Max-Age': '86400'
  };
}

/**
 * Get all headers combined
 * @param {object} options
 * @returns {object}
 */
export function getAllHeaders({ isContainer = false, etag = null, contentType = null, origin = null, resourceUrl = null, wacAllow = null, connegEnabled = false, updatesVia = null }) {
  return {
    ...getResponseHeaders({ isContainer, etag, contentType, resourceUrl, wacAllow, connegEnabled, updatesVia }),
    ...getCorsHeaders(origin)
  };
}

/**
 * Get headers for 404 responses (non-existent resources)
 * These headers tell clients what methods are supported for creating the resource
 * @param {object} options
 * @returns {object}
 */
export function getNotFoundHeaders({ resourceUrl = null, origin = null, connegEnabled = false }) {
  // Determine if this would be a container based on URL ending with /
  const isContainer = resourceUrl?.endsWith('/') || false;
  const aclUrl = resourceUrl ? getAclUrl(resourceUrl, isContainer) : null;

  // Get Accept-* headers
  const acceptHeaders = getAcceptHeaders(connegEnabled, isContainer);

  const headers = {
    ...getCorsHeaders(origin),
    'Link': aclUrl ? `<${aclUrl}>; rel="acl"` : '',
    'Accept-Patch': 'text/n3, application/sparql-update',
    'Accept-Put': acceptHeaders['Accept-Put'] || 'application/ld+json, */*',
    'Allow': 'GET, HEAD, PUT, PATCH, OPTIONS' + (isContainer ? ', POST' : ''),
    'Vary': connegEnabled ? 'Accept, Authorization, Origin' : 'Authorization, Origin'
  };

  if (isContainer && acceptHeaders['Accept-Post']) {
    headers['Accept-Post'] = acceptHeaders['Accept-Post'];
  }

  return headers;
}
