import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MetodologiaCRISPComponent } from './metodologia-crisp.component';

describe('MetodologiaCRISPComponent', () => {
  let component: MetodologiaCRISPComponent;
  let fixture: ComponentFixture<MetodologiaCRISPComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ MetodologiaCRISPComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(MetodologiaCRISPComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
