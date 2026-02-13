const { z } = require("zod");

const InsightsSchema = z.object({
  summary: z.string().min(20),

  keyConcepts: z
    .array(z.string())
    .min(1)
    .max(6),

  flashcards: z
    .array(
      z.object({
        question: z.string().min(5),
        answer: z.string().min(5),
      })
    )
    .max(5),

  actionItems: z
    .array(z.string())
    .min(1)
    .max(5),
});

module.exports = {
  InsightsSchema,
};
