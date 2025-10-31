// Quick schema check utility
use rusqlite::{Connection, Result};

fn main() -> Result<()> {
    let conn = Connection::open("/app/data/ontology.db")?;

    println!("=== owl_classes table schema ===");
    let mut stmt = conn.prepare("PRAGMA table_info(owl_classes)")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i32>(0)?,  // cid
            row.get::<_, String>(1)?,  // name
            row.get::<_, String>(2)?,  // type
            row.get::<_, i32>(3)?,  // notnull
            row.get::<_, Option<String>>(4)?,  // dflt_value
            row.get::<_, i32>(5)?,  // pk
        ))
    })?;

    for row in rows {
        let (cid, name, type_, notnull, dflt, pk) = row?;
        println!("Column {}: {} ({}) notnull={} default={:?} pk={}",
            cid, name, type_, notnull, dflt, pk);
    }

    Ok(())
}
