module.exports = {
  beforeRequest: (requestParams, context, ee, next) => {
    // Add custom headers or modify request
    return next();
  },
  afterResponse: (requestParams, response, context, ee, next) => {
    // Process response, extract data, etc.
    return next();
  }
};
