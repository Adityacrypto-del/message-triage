# data/

The evaluation dataset is **not** committed here. The assignment brief states
the supplied CSV must not be published to a public repository, so `.gitignore`
excludes everything in this folder except this file.

To run the project locally, drop these two files in:

```
data/
├── messages.csv            # message_id, timestamp, sender, message
└── mandatory_demo_ids.csv  # message_id
```

Then:

```bash
python run_pipeline.py --data data --out outputs
```

## Expected schema

`messages.csv`

| column       | type   | notes                                    |
|--------------|--------|------------------------------------------|
| `message_id` | string | unique, e.g. `MSG_0001`                  |
| `timestamp`  | string | `YYYY-MM-DD HH:MM:SS`                    |
| `sender`     | string | display name                             |
| `message`    | string | free text                                |

Rows whose timestamp cannot be parsed are kept in file order rather than
dropped or assigned an invented date; the loader reports how many that was.
