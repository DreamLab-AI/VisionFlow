using System.Collections.Generic;
using NUnit.Framework;
using VisionFlow.Network;

namespace VisionFlow.Tests.Editor
{
    public sealed class NostrEventTests
    {
        [Test]
        public void CollectReferencedNodeIds_GathersTagValues()
        {
            var ev = new NostrEvent
            {
                Tags = new List<List<string>>
                {
                    new() { "e",    "event-ref" },
                    new() { "node", "node-42"   },
                    new() { "p",    "pubkey"    },
                },
            };

            var ids = new List<string>();
            ev.CollectReferencedNodeIds(ids);

            Assert.Contains("event-ref", ids);
            Assert.Contains("node-42", ids);
            Assert.AreEqual(2, ids.Count);
        }
    }
}
