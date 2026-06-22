using NUnit.Framework;
using VisionFlow.Network;

namespace VisionFlow.Tests.Editor
{
    public sealed class GovernanceRequestTests
    {
        [Test]
        public void TryFromNostrEvent_AcceptsKind31402WithTags()
        {
            var ev = new NostrEvent
            {
                Id         = "abc123",
                Pubkey     = "deadbeef",
                Kind       = 31402,
                CreatedAt  = 1_700_000_000,
                Content    = "spawn sub-agent",
                Tags = new System.Collections.Generic.List<System.Collections.Generic.List<string>>
                {
                    new() { "agent",  "did:nostr:agent1" },
                    new() { "action", "execute_tool"     },
                },
            };

            Assert.IsTrue(GovernanceRequest.TryFromNostrEvent(ev, out var req));
            Assert.AreEqual("abc123", req.EventId);
            Assert.AreEqual("did:nostr:agent1", req.AgentDid);
            Assert.AreEqual("execute_tool", req.Action);
            Assert.AreEqual("spawn sub-agent", req.Context);
        }

        [Test]
        public void TryFromNostrEvent_RejectsOtherKinds()
        {
            var ev = new NostrEvent { Id = "x", Kind = 31403 };
            Assert.IsFalse(GovernanceRequest.TryFromNostrEvent(ev, out _));
        }
    }
}
